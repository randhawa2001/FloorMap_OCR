import cv2
import easyocr
import re
import json


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, image

def extract_text_easyocr(image):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image, detail=1, paragraph=False)
    return results

def find_dimensions_easyocr(results):

    dimension_pattern = re.compile(
        r'\(?\d+\'-?\d*"?\s*[Xx×]\s*\d+\'-?\d*"?\)?|\(?\d+"?-?\d*"?\s*[Xx×]\s*\d+"?-?\d*"?\)?')

    dimensions = [text for _, text, prob in results if dimension_pattern.search(text) and prob > 0.3]
    return dimensions

def draw_bounding_boxes_easyocr(image, results):
    dimension_pattern = re.compile(
        r'\(?\d+\'-?\d*"?\s*[Xx×]\s*\d+\'-?\d*"?\)?|\(?\d+"?-?\d*"?\s*[Xx×]\s*\d+"?-?\d*"?\)?')
    for (bbox, text, prob) in results:
        if dimension_pattern.search(text) and prob > 0.3:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image

def main_easyocr(image_path):
    processed_image, original_image = preprocess_image(image_path)
    results = extract_text_easyocr(original_image)

    print("Raw OCR Results:")
    for bbox, text, prob in results:
        print(f"Detected Text: '{text}' with Confidence: {prob}")

    dimensions = find_dimensions_easyocr(results)
    if not dimensions:
        print("No dimensions detected. Check the raw OCR results above.")

    boxed_image = draw_bounding_boxes_easyocr(original_image, results)

    print("Extracted Dimensions:", json.dumps(dimensions, indent=2))
    cv2.imshow('Detected Dimensions', boxed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = 'test_1.png'
    main_easyocr(image_path)
