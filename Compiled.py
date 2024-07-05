import cv2
import numpy as np
from keras.src.saving import load_model
from PIL import Image

# Loading the Pretrained Model
model_path = '../Working_Files/model_1.h5'
model = load_model(model_path)

# Function to prepare single cell to be fed to the model
def Prep(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #making image grayscale
    img = cv2.equalizeHist(img) #Histogram equalization to enhance contrast
    img = img/255 #normalizing
    return img

# Function to predict the number on the image
def predict_(img):
    img = cv2.resize(img, (32, 32))

    img = Prep(img)
    img = np.array(img)

    img = img.reshape(1, 32, 32, 1)
    pred = model.predict(img)
    output = pred.argmax()

    return output, pred.argmax()

# function to greyscale, blur and change the receptive threshold of image
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3),6)
    threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    return threshold_img


# Functions to extract individual cells from the sudoku image
def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest ,max_area

def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def process_sudoku(su_puzzle):
    # Finding the outline of the sudoku puzzle in the image
    su_contour_1= su_puzzle.copy()
    su_contour_2= su_puzzle.copy()
    su_contour, hierarchy = cv2.findContours(su_puzzle,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(su_contour_1, su_contour,-1,(0,255,0),3)

    black_img = np.zeros((450,450,3), np.uint8)
    su_biggest, su_maxArea = main_outline(su_contour)
    if su_biggest.size != 0:
        su_biggest = reframe(su_biggest)
        cv2.drawContours(su_contour_2,su_biggest,-1, (0,255,0),10)
        su_pts1 = np.float32(su_biggest)
        su_pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
        su_matrix = cv2.getPerspectiveTransform(su_pts1,su_pts2)
        su_imagewrap = cv2.warpPerspective(puzzle,su_matrix,(450,450))
        su_imagewrap =cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)
    return su_imagewrap

def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


def CropCell(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        # print(img.shape)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)

    return Cells_croped


def read_cells(cell, model):
    result = []
    for img in cell:
        # preprocess the image as it was in the model
        img = np.asarray(img)
        # img = clean_image(img)
        # print(img.shape)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        # plt.imshow(img)
        # plt.show()
        img = img.reshape(1, 32, 32, 1)

        # getting predictions and setting the values if probabilities are above 75%

        predictions = model.predict(img, verbose = 0)
        classIndex = np.argmax(predictions, axis=1)
        probabilityValue = np.amax(predictions)

        # print(classIndex)
        if probabilityValue > 0.50:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


def is_valid(board, row, col, num):
    # Check if 'num' is not in the current row
    if num in board[row, :]:
        return False
    # Check if 'num' is not in the current column
    if num in board[:, col]:
        return False
    # Check if 'num' is not in the current 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row + 3, start_col:start_col + 3]:
        return False
    return True


def solve_sudoku(board):
    empty = np.where(board == 0)
    if len(empty[0]) == 0:  # No empty cells, puzzle solved
        return True

    row, col = empty[0][0], empty[1][0]

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num
            if solve_sudoku(board):
                return True
            board[row, col] = 0  # Reset cell and backtrack

    return False

def print_sudoku(quiz):
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("....................")

        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")

            if col == 8:
                print(quiz[row][col])
            else:
                print(str(quiz[row][col]) + " ", end="")


# Input Your image
image_path = './Sudoku_5.jpg'
puzzle = cv2.imread(image_path)

# Resizing puzzle to be solved
puzzle = cv2.resize(puzzle, (450,450))

# Preprocessing Puzzle
su_puzzle = preprocess(puzzle)

# Processing the Sudoku
su_imagewrap = process_sudoku(su_puzzle)

# Splitting the cells
sudoku_cell = splitcells(su_imagewrap)

# Croping the Cells
sudoku_cell_croped= CropCell(sudoku_cell)

# Creating Grid of the sudoku
grid = read_cells(sudoku_cell_croped, model)
unsolved = np.reshape(grid, (9,9))

print("\n Detected Sudoku : \n")
print_sudoku(unsolved)

grid = np.asarray(grid)
grid = np.reshape(grid, (9,9))

if solve_sudoku(grid):
    print("\n Sudoku solved successfully! \n")
    print_sudoku(grid)
else:
    print("No solution exists.")
