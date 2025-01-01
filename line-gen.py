import cv2
import numpy as np


def detect_walls_and_rooms(image_path, output_path='walls_and_rooms.png'):
    #Image loading....
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)


    wall_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(wall_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    room_image = image.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            cv2.drawContours(room_image, [contour], -1, (255, 0, 0), 2)


    combined_image = cv2.addWeighted(room_image, 0.8, wall_image, 1, 0)

    #
    cv2.imwrite(output_path, combined_image)
    cv2.imshow('Walls and Rooms', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Path to the uploaded image
image_path = 'test_1.png'
detect_walls_and_rooms(image_path)
