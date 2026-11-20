
# Importing the necessary libraries
import cv2
import numpy as np
import gradio as gr
import onnxruntime as ort

# Loading our trained CNN model
session = ort.InferenceSession(
    "TrainedModel/best.onnx",
    providers=["CPUExecutionProvider"]
)

inputName = session.get_inputs()[0].name
outputName = session.get_outputs()[0].name

# Processing images uploaded through the web interface
def processInventory(uploadedImage):

    # Checking whether an image was uploaded
    if uploadedImage is None:
        return None, 0

    # Resizing the image while preserving its aspect ratio
    originalHeight, originalWidth = uploadedImage.shape[:2]
    scale = min(640 / originalWidth, 640 / originalHeight)

    newWidth = round(originalWidth * scale)
    newHeight = round(originalHeight * scale)

    resizedImage = cv2.resize(uploadedImage, (newWidth, newHeight))

    # Padding the image to the model's input size
    padWidth = 640 - newWidth
    padHeight = 640 - newHeight

    left = padWidth // 2
    top = padHeight // 2

    paddedImage = cv2.copyMakeBorder(
        resizedImage,
        top,
        padHeight - top,
        left,
        padWidth - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    # Preparing the image for the CNN
    inputImage = paddedImage.astype(np.float32) / 255.0
    inputImage = np.transpose(inputImage, (2, 0, 1))
    inputImage = np.expand_dims(inputImage, axis=0)

    # Detecting soda cans using our trained CNN
    predictions = session.run(
        [outputName],
        {inputName: inputImage}
    )[0]

    # Processing the model's predictions
    predictions = predictions[0].T

    boxes = []
    scores = []

    for prediction in predictions:
        classScores = prediction[4:]
        confidence = float(np.max(classScores))

        if confidence >= 0.25:
            x, y, w, h = prediction[:4]

            x1 = (x - w / 2 - left) / scale
            y1 = (y - h / 2 - top) / scale
            x2 = (x + w / 2 - left) / scale
            y2 = (y + h / 2 - top) / scale

            x1 = max(0, min(originalWidth - 1, int(x1)))
            y1 = max(0, min(originalHeight - 1, int(y1)))
            x2 = max(0, min(originalWidth - 1, int(x2)))
            y2 = max(0, min(originalHeight - 1, int(y2)))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(confidence)

    # Removing overlapping detections
    selectedIndices = cv2.dnn.NMSBoxes(
        boxes, scores, 0.25, 0.40
    )

    # Creating a copy of the uploaded image
    resultImage = uploadedImage.copy()

    # Drawing bounding boxes and numbering inventory items
    itemNumber = 1

    for index in np.array(selectedIndices).flatten():
        x, y, w, h = boxes[int(index)]

        cv2.rectangle(
            resultImage,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            resultImage,
            str(itemNumber),
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        itemNumber += 1

    # Calculating the total inventory count
    totalCount = itemNumber - 1

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

