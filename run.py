import sys
import numpy as np
from PIL import Image
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                             QSlider, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

SIZE = 800


def apply_salt_and_pepper(image: np.ndarray, noise_density: float) -> np.ndarray:
    output = np.copy(image)
    # Generate salt noise
    salt = np.random.random(image.shape) < (noise_density / 2)
    output[salt] = 255
    # Generate pepper noise
    pepper = np.random.random(image.shape) < (noise_density / 2)
    output[pepper] = 0
    return output


def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    magnitude = abs(sobel_x) + abs(sobel_y)
    # Normalize to 0-255
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return magnitude


class ImageProcessor(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, image, noise_level):
        super().__init__()
        self.image = image
        self.noise_level = noise_level

    def run(self):
        # Apply noise
        self.progress.emit(33)
        noisy_image = apply_salt_and_pepper(self.image, self.noise_level)

        # Edge detection
        self.progress.emit(66)
        edge_image = sobel_edge_detection(self.image)

        self.progress.emit(100)
        self.finished.emit(noisy_image, edge_image)


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.noise_slider = None
        self.noise_label = None
        self.progress_bar = None
        self.open_button = None
        self.edge_display = None
        self.edge_label = None
        self.original_label = None
        self.noisy_label = None
        self.original_display = None
        self.noisy_display = None
        self.processor = None
        self.init_ui()
        self.current_image = None

    def init_ui(self) -> None:
        self.setWindowTitle('Image Processing Demo')
        self.setGeometry(100, 100, 1200, 600)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Create image display area
        image_layout = QHBoxLayout()

        # Original image
        self.original_label = QLabel('Original Image')
        self.original_display = QLabel()
        self.original_display.setFixedSize(SIZE, SIZE)
        self.original_display.setAlignment(Qt.AlignCenter)
        self.original_display.setStyleSheet("border: 1px solid black")

        # Noisy image
        self.noisy_label = QLabel('Noisy Image')
        self.noisy_display = QLabel()
        self.noisy_display.setFixedSize(SIZE, SIZE)
        self.noisy_display.setAlignment(Qt.AlignCenter)
        self.noisy_display.setStyleSheet("border: 1px solid black")

        # Edge detection image
        self.edge_label = QLabel('Edge Detection')
        self.edge_display = QLabel()
        self.edge_display.setFixedSize(SIZE, SIZE)
        self.edge_display.setAlignment(Qt.AlignCenter)
        self.edge_display.setStyleSheet("border: 1px solid black")

        # Add displays to layout
        for label, display in [(self.original_label, self.original_display),
                               (self.noisy_label, self.noisy_display),
                               (self.edge_label, self.edge_display)]:
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.addWidget(label, alignment=Qt.AlignCenter)
            container_layout.addWidget(display)
            container.setLayout(container_layout)
            image_layout.addWidget(container)

        layout.addLayout(image_layout)

        # Controls
        controls_layout = QHBoxLayout()

        # Open image button
        self.open_button = QPushButton('Open Image')
        self.open_button.clicked.connect(self.open_image)
        controls_layout.addWidget(self.open_button)

        # Noise level slider
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(1)
        self.noise_slider.setMaximum(20)
        self.noise_slider.setValue(5)
        self.noise_slider.valueChanged.connect(self.noise_level_changed)
        controls_layout.addWidget(self.noise_slider)

        self.noise_label = QLabel('Noise Level: 5%')
        controls_layout.addWidget(self.noise_label)

        layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        main_widget.setLayout(layout)

    def open_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_name:
            try:
                # Load image using PIL and convert to numpy array
                image = np.array(Image.open(file_name))
                self.current_image = image

                # Display original image
                self.display_image(image, self.original_display)

                # Process image
                self.process_image()
            except Exception as e:
                print(f"Error loading image: {e}")

    def noise_level_changed(self, value: float | int) -> None:
        self.noise_label.setText(f'Noise Level: {value}%')
        if self.current_image is not None:
            self.process_image()

    def process_image(self) -> None:
        self.progress_bar.show()
        self.open_button.setEnabled(False)
        self.noise_slider.setEnabled(False)

        # Create and start processor thread
        self.processor = ImageProcessor(self.current_image, self.noise_slider.value() / 100)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.finished.connect(self.update_processed_images)
        self.processor.start()

    def update_processed_images(self, noisy_image: np.ndarray, edge_image: np.ndarray) -> None:
        self.display_image(noisy_image, self.noisy_display)
        self.display_image(edge_image, self.edge_display)

        self.progress_bar.hide()
        self.open_button.setEnabled(True)
        self.noise_slider.setEnabled(True)

    def display_image(self, image: Image, label: QLabel) -> None:
        # Convert numpy array to QPixmap and display
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line,
                             QImage.Format_RGB888)
        else:
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line,
                             QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    ex.show()
    sys.exit(app.exec_())
