import cv2

# Callback function to print coordinates on left mouse click
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Target Coordinate: (X: {x}, Y: {y})")

# Load your exported Canva template
image_path = 'data/synthetic/base_template_v1.png' 
img = cv2.imread(image_path)

if img is None:
    print("Error: Could not load image. Check the path.")
else:
    print("Click on the image to get (X, Y) coordinates. Press 'q' or 'ESC' to quit.")
    cv2.imshow('Coordinate Mapper', img)
    cv2.setMouseCallback('Coordinate Mapper', get_coordinates)
    
    # Wait for the user to press 'q' or 'Esc' to close
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()