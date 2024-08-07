import fitz
import logging
from unstructured.partition.pdf import partition_pdf


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


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
        logger.info(f'Total no of pages: {str(page_nos)}.')
        if page_nos < 80:
            elements = partition_pdf(
                filename=pdf_file,
                strategy="hi_res"
            )
            content = ''
            for element in elements:
                if element.category == 'Image':
                    image_text = element.text
                    if image_text:
                        content += f"\n\n{str(image_text)}"
                if element.category not in ['Image', 'Table', 'Footer', 'Header', 'FigureCaption']:
                    content += f"\n\n{str(element)}"
        else:
            logger.info('Page number > 80...')
            content = extract_using_pymupdf(pdf_file)
        return content
    except Exception as e:
        raise Exception(e)


def extract_by_components(pdf_file):
    try:
        elements = partition_pdf(
            filename=pdf_file,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Table", "Image"],
            extract_image_block_to_payload=True
        )
        all_components = []
        sequence_id = 1
        for element in elements:
            if element.category in ['Image', 'Table']:
                image_base64 = element.metadata.image_base64
                all_components.append({'sequence_id': sequence_id,
                                       'content_type': 'image',
                                       'disclosure_content_image': image_base64
                                       }
                                      )
            else:
                all_components.append({'sequence_id': sequence_id,
                                       'content_type': 'text',
                                       'content_text': element.text
                                       }
                                      )
            sequence_id += 1
        return all_components
    except Exception as e:
        raise Exception(e)
