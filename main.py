import numpy as np
import cv2
import streamlit as st

def barycentric_interpolation(image, zoom_factor):
    image = image.astype(np.float64)  # Convert image to float64

    # Compute the dimensions of the zoomed image
    zoomed_height = int(image.shape[0] * zoom_factor)
    zoomed_width = int(image.shape[1] * zoom_factor)

    # Create an empty zoomed image with the new dimensions
    zoomed_image = np.zeros((zoomed_height, zoomed_width, 3), dtype=np.uint8)

    # Compute the scale factors for row and column
    scale_row = (image.shape[0] - 1) / (zoomed_height - 1)
    scale_col = (image.shape[1] - 1) / (zoomed_width - 1)

    # Compute the zoomed image using barycentric interpolation
    for row in range(zoomed_height):
        for col in range(zoomed_width):
            original_row = row * scale_row
            original_col = col * scale_col

            # Compute the integer and fractional parts of the row and column
            int_row = int(original_row)
            frac_row = original_row - int_row
            int_col = int(original_col)
            frac_col = original_col - int_col

            # Perform barycentric interpolation on each channel
            for channel in range(3):
                q01 = image[max(int_row - 1, 0), int_col, channel]

                q11 = image[int_row, int_col, channel]

                q21 = image[min(int_row + 1, image.shape[0] - 1), int_col, channel]

                q31 = image[min(int_row + 2, image.shape[0] - 1), int_col, channel]

                # Perform barycentric interpolation in both dimensions
                p2 = (1 - frac_row) * (q01 - q11) + q11
                p3 = frac_row * (frac_col * (q31 - q21) + q21 - (frac_col * (q11 - q31) + q31)) + (frac_col * (q11 - q31) + q31)

                r0 = p2
                r1 = frac_col * (p3 - p2) + p2

                # Assign the interpolated value to the zoomed image
                zoomed_image[row, col, channel] = np.round((r1 - r0) * frac_col + r0)

    return zoomed_image


# Streamlit App
def main():
    st.title('Barycentric Interpolation for Image Zooming')
    st.subheader("Author : Luka Gorgadze")
    st.write('Barycentric interpolation is a method to perform image zooming. This app allows you to upload your image and zoom it using barycentric interpolation.')

    # Upload the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the original image
        st.subheader('Original Image')
        st.image(image, channels="BGR")

        # Set the zoom factor using a slider
        zoom_factor = st.slider("Zoom Factor", 1.0, 3.0, 2.0, 0.1)

        # Perform zooming using barycentric interpolation
        zoomed_image = barycentric_interpolation(image, zoom_factor)

        # Display the zoomed image
        st.subheader(f'Zoomed Image (Zoom Factor: {zoom_factor})')
        st.image(zoomed_image, channels="BGR")

if __name__ == "__main__":
    main()