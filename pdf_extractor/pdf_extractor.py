import fitz
import logging
from unstructured.partition.pdf import partition_pdf
import openai
import boto3
from botocore.exceptions import NoCredentialsError
import pytesseract
from PIL import Image
import os
from base64 import b64decode
from io import BytesIO
from dotenv import load_dotenv
import requests

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
# load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
local_llm_url = os.environ.get("LOCAL_LLM_URL")
table_extraction_prompt = """Below is the result of an extracted text from a table image:
    {content}

    # Instruction
    1. Extract meaningful summary and do not add any additional information or make up any information.
    2. IMPORTANT: the information should be conveyed correctly. If not, it will give a wrong interpretation. Make sure to double check.
"""
use_gpt_as_fallback = os.environ.get("USE_GPT_AS_FALLBACK", "false").lower() == "true"

def call_local_llm(content, model_name="llama3.1:latest"):
    prompt = table_extraction_prompt.format(content=content)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 10000},
    }
    try:
        response = requests.post(local_llm_url, json=payload, timeout=120)

        response_json = response.json()

        response_text = response_json.get("response")

        return response_text
    except requests.exceptions.Timeout:
        raise Exception("The request timed out after 120 seconds.")


def generate_image_summary(bucket_name, key):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"https://{bucket_name}.s3.{os.environ.get('AWS_REGION')}.amazonaws.com/{key}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    summary = response.choices[0].message.content
    return summary


def upload_to_s3(local_file, bucket_name, s3_file):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        s3.upload_file(local_file, bucket_name, s3_file)
        print(f"Upload Successful: {s3_file}")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def extract_text_using_pytesseract(image_file):
    # Load the image from a local path
    image = Image.open(image_file)

    # Extract text from the image using pytesseract
    extracted_text = pytesseract.image_to_string(image)
    if not extracted_text or extracted_text == "\x0c":
        return None
    return extracted_text

def extract_using_pymupdf(temp_pdf):
    logger.info(f"Extracting using pymupdf: {temp_pdf}")
    doc = fitz.open(temp_pdf)
    texts = []
    for page in doc:
        text = page.get_text()
        if text:
            texts.append(text)

    content = ",".join(texts).replace("'", "")
    if not content:
        logger.error(f"Failed to extract {temp_pdf}")
    doc.close()
    return content


def count_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return doc.page_count


def extract_whole_content(pdf_file):
    try:
        page_nos = count_pages(pdf_file)
        logger.info(f"Total no of pages: {str(page_nos)}.")
        if page_nos < 80:
            elements = partition_pdf(
                filename=pdf_file,
                strategy="hi_res",
                hi_res_model_name="yolox",
                extract_images_in_pdf=True,
                extract_image_block_types=["Table", "Image"],
                extract_image_block_to_payload=True,
            )
            content = ""
            logger.info(f'Extracted all the elements of length: {len(elements)}')
            for element in elements:
                if element.category == "Table":
                    if element.metadata.image_base64:

                        # Decode the base64 image
                        image_data = b64decode(element.metadata.image_base64)
                        image = Image.open(BytesIO(image_data))

                        # Create a unique local file name
                        local_image_file = "pdf-temp-image.jpg"

                        # Save the image locally as a .jpg file
                        image.save(local_image_file, format="JPEG")

                        summary = None
                        try:
                            image_text = extract_text_using_pytesseract(local_image_file)
                            if image_text:
                                summary = call_local_llm(image_text)
                            elif image_text is None and use_gpt_as_fallback:
                                # Upload to S3
                                upload_to_s3(
                                    local_image_file,
                                    os.environ.get("PUBLIC_S3_BUCKET"),
                                    "temp_images/pdf-temp-image.jpg",
                                )
                                summary = generate_image_summary(
                                    os.environ.get("PUBLIC_S3_BUCKET"),
                                    "temp_images/pdf-temp-image.jpg",
                                )
                            else:
                                summary = ''
                        except Exception as e:
                            logger.error(
                                f"Exception occurred while extracting table: {e}"
                            )
                        if summary:
                            content += f"\n\n{str(summary)}"
                        os.remove(local_image_file)
                if element.category == "Image":
                    image_text = element.text
                    if image_text:
                        content += f"\n\n{str(image_text)}"
                if element.category not in [
                    "Image",
                    "Table",
                    "Footer",
                    "Header",
                    "FigureCaption",
                ]:
                    content += f"\n\n{str(element)}"
        else:
            logger.info("Page number > 80...")
            content = extract_using_pymupdf(pdf_file)
        return content
    except Exception as e:
        logger.error(f'Exception occurred while extracting content:{str(e)}')
        raise Exception(e)


def extract_by_components(pdf_file):
    try:
        elements = partition_pdf(
            filename=pdf_file,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Table", "Image"],
            extract_image_block_to_payload=True,
        )
        all_components = []
        sequence_id = 1
        for element in elements:
            if element.category in ["Image", "Table"]:
                image_base64 = element.metadata.image_base64
                all_components.append(
                    {
                        "sequence_id": sequence_id,
                        "content_type": "image",
                        "disclosure_content_image": image_base64,
                    }
                )
            else:
                all_components.append(
                    {
                        "sequence_id": sequence_id,
                        "content_type": "text",
                        "content_text": element.text,
                    }
                )
            sequence_id += 1
        return all_components
    except Exception as e:
        raise Exception(e)


extract_whole_content('/home/sarita/Downloads/02200521.pdf')