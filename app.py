# Importing the necessary libraries
import cv2
import numpy as np
import gradio as gr

# Processing images uploaded through the web interface
def processInventory(uploadedImage):

    # Checking whether an image was uploaded
    if uploadedImage is None:
        return None, 0

    # Converting the uploaded image to OpenCV's BGR format
    image = cv2.cvtColor(uploadedImage, cv2.COLOR_RGB2BGR)

    # Converting the image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian blurring
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

    # Applying Canny edge detection
    edgeImage = cv2.Canny(blurredImage, 50, 150)

    # Applying morphological closing
    kernel = np.ones((5, 5), np.uint8)
    closedImage = cv2.morphologyEx(edgeImage, cv2.MORPH_CLOSE, kernel)

    # Finding the external contours
    contours, hierarchy = cv2.findContours(closedImage,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Filtering out small contours
    filteredContours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= 1000:
            filteredContours.append(contour)

    # Creating a copy of the uploaded image
    resultImage = uploadedImage.copy()

    # Drawing bounding boxes and numbering inventory items
    itemNumber = 1

    for contour in filteredContours:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(resultImage, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        cv2.putText(resultImage, str(itemNumber), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        itemNumber += 1

    # Calculating the total inventory count
    totalCount = len(filteredContours)

    return resultImage, totalCount


# Creating the inventory counting web interface
with gr.Blocks(title="Inventory Counting System") as demo:

    gr.Markdown(
        "Upload a photograph of your inventory, then click **Count Inventory** "
        "to automatically detect and count the items. "
        "The results will display numbered bounding boxes around detected items "
        "and the total inventory count. "
        "Click **Clear** to reset the interface and upload another photograph."
    )

    # Creating the image upload and results display
    with gr.Row():
        inputImage = gr.Image(label="Upload Inventory Image", type="numpy")
        outputImage = gr.Image(label="Detected Inventory Items")

    # Displaying the total inventory count
    outputCount = gr.Number(label="Total Inventory Count", precision=0)

    # Creating the button to process the uploaded image
    countButton = gr.Button("Count Inventory")
    
    # Creating a button to clear the uploaded image and results
    clearButton = gr.ClearButton(
        components=[inputImage, outputImage, outputCount],
        value="Clear"
    )

    # Connecting the button to the inventory counting function
    countButton.click(
        fn=processInventory,
        inputs=inputImage,
        outputs=[outputImage, outputCount]
    )

# Launching the web interface
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
