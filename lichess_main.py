import cv2
import numpy as np
from PIL import ImageGrab
import time
import imutils
import pyautogui as ptg
from skimage.metrics import structural_similarity as compare_ssim
import chess
import chess.engine
import asyncio
from random import randint
import subprocess

'''
Made by Ryan Lee In Zer

This script is specifically designed for https://chess.com, on Windows 10

PLEASE DO NOT USE THIS SCRIPT FOR RATED GAMES

This script is made for the purpose purely for coding practice and learning image processing,
I do not condone using this script for malicious purposes such as cheating and/or gaining more rating.
I am not responsible for any consequences led by the usage of this script.
I encourage the usage of this bot to play against another bot.
'''

#bounding area of the chess board (top left x, top left y, bottom right x, bottom right y)
bbox = (601,201,1225, 822) #lichess
length = (bbox[2] - bbox[0])//8 #length of each square 

#get the board position based on input
def get_board_coordinate(x,y,opponent_color):
    rows = ['8','7','6','5','4','3','2','1']
    columns = ['a','b','c','d','e','f','g','h']

    if opponent_color == "white":
        rows = rows[::-1]
        columns = columns[::-1]

    board_x = columns[round((x - length//2)/length)]
    board_y = rows[round((y - length//2)/length)]

    return f'{board_x}{board_y}'

#detect if piece is on check
def detect_check(x,y,image):
    dist = length//2

    starting_x = x - dist
    ending_x = x + dist
    starting_y = y - dist
    ending_y = y + dist

    if starting_x < 0:
        starting_x = 0
    if starting_y < 0:
        starting_y = 0
    if ending_x > bbox[2]-bbox[0]:
        ending_x = bbox[2]-bbox[0]
    if ending_y > bbox[3]-bbox[1]:
        ending_y = bbox[3]-bbox[1]

    img = image[starting_y:ending_y,starting_x:ending_x]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    mask = cv2.bitwise_or(mask1, mask2)
    return (cv2.countNonZero(mask) > 10)

#determine if it's black or white color chess piece
def determine_color(x,y,image):
    dist = length//2

    starting_x = x - dist
    ending_x = x + dist
    starting_y = y - dist
    ending_y = y + dist

    if starting_x < 0:
        starting_x = 0
    if starting_y < 0:
        starting_y = 0
    if ending_x > bbox[2]-bbox[0]:
        ending_x = bbox[2]-bbox[0]
    if ending_y > bbox[3]-bbox[1]:
        ending_y = bbox[3]-bbox[1]

    img = image[starting_y:ending_y,starting_x:ending_x]
    
    white_pixels = np.count_nonzero((img >= [245, 245, 245]).all(axis = 2))
    black_pixels = np.count_nonzero((img <= [88,85,84]).all(axis = 2))

    if white_pixels > 500:
        return "white"
    if black_pixels > 500:
        return "black"
    return None

#waiting for opponent to make a move before capturing a second image
def capturing_second_image(image1):
    while 1:
        image2 = np.array(ImageGrab.grab(bbox=bbox))
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        if not(np.bitwise_xor(image1,image2).any()):
            continue
        else:
            time.sleep(0.5)
            break
    image2 = np.array(ImageGrab.grab(bbox=bbox))
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    return image2

#figure out what colour player (me) is playing (black or white)
def playing_color():
    global length
    image = np.array(ImageGrab.grab(bbox=bbox))
    play_color = determine_color(length//2,(bbox[3]-bbox[1])-length//2,image)
    opponent_color = "black" if play_color == 'white' else 'white'

    return play_color, opponent_color

#play out the move the AI suggested on the chess board
def play_move(player_color,before, after, is_pawn):
    before_column = before[0]
    before_row = before[1]

    after_column = after[0]
    after_row = after[1]

    rows = ['8','7','6','5','4','3','2','1']
    columns = ['a','b','c','d','e','f','g','h']

    if player_color == "black":
        rows = rows[::-1]
        columns = columns[::-1]

    before_x_index = columns.index(before_column)
    before_y_index = rows.index(before_row)

    after_x_index = columns.index(after_column)
    after_y_index = rows.index(after_row)

    before_x = bbox[0] + length//2 +  length * before_x_index
    before_y = bbox[1] + length//2 +  length * before_y_index

    after_x = bbox[0] + length//2 +  length * after_x_index
    after_y = bbox[1] + length//2 +  length * after_y_index

    ptg.moveTo(before_x, before_y)
    time.sleep(0.2)
    ptg.dragTo(after_x, after_y, 0.1,button='left')
    if ((after_row == '8' and player_color == 'white' and before_row == '7') or (after_row == '1' and player_color == 'black' and before_row == '2')) and abs(before_x_index - after_x_index) <= 1:
        if is_pawn:
            ptg.click()

#correctly identify movement of chess pieces when castling
def on_castle(array, opponent_color):
    queen_side = False
    king_side = False
    if opponent_color == 'black':
        for c in array:
            if c[0] == 'h8':
                king_side = True
            if c[0] == 'a8':
                queen_side = True
    elif opponent_color == 'white':
        for c in array:
            if c[0] == 'h1':
                king_side = True
            if c[0] == 'a1':
                queen_side = True
        
    
    new_array = array[:]
    if king_side and opponent_color == 'black':
        for index, array1 in enumerate(new_array):
            if array1[0] != 'e8' and array1[0] != 'g8':
                new_array[index] = []
    
    elif queen_side and opponent_color == 'black':
        for index, array1 in enumerate(new_array):
            if array1[0] != 'e8' and array1[0] != 'c8':
                new_array[index] = []
    
    elif king_side and opponent_color == 'white':
        for index, array1 in enumerate(new_array):
            if array1[0] != 'e1' and array1[0] != 'g1':
                new_array[index] = []
    elif queen_side and opponent_color == 'white':
        for index, array1 in enumerate(new_array):
            if array1[0] != 'e1' and array1[0] != 'c1':
                new_array[index] = []
    
    while [] in new_array: new_array.remove([])

    return new_array

#correctly adjust captured coordinates
def fix_coord(pos, opponent_color):
    rows = ['8','7','6','5','4','3','2','1']
    columns = ['a','b','c','d','e','f','g','h']

    if opponent_color == 'white':
        rows = rows[::-1]
        columns = columns[::-1]
    
    x = columns.index(pos[0]) * length + length // 2
    y = rows.index(pos[1]) * length +  length // 2

    return x,y

#determine the movement of the opponent's chess piece
def determine_opponent_move(imageA, imageB, opponent_color):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    score, diff = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    im_floodfill = thresh.copy()

    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = thresh | im_floodfill_inv

    cv2.imwrite('thresh.png', im_out)

    cnts = cv2.findContours(im_out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    all_coord = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cX = x + w//2
        cY = y + h//2
        if detect_check(cX,cY,imageA):
            all_coord.append([get_board_coordinate(cX,cY,opponent_color),True,[cX,cY]])
        else:
            all_coord.append([get_board_coordinate(cX,cY,opponent_color), False,[cX,cY]])

    last_board_coordinate = ''
    all_coord.sort(key=lambda x:x[0])
    for i, coord in enumerate(all_coord):
        if coord[0] == last_board_coordinate:
            all_coord[i] = []
        else:
            last_board_coordinate = coord[0]
    
    while [] in all_coord: all_coord.remove([])

    if len(all_coord) >= 3:
        for i, c in enumerate(all_coord):
            if c[1] is True:
                all_coord[i] = []
    while [] in all_coord: all_coord.remove([])
    print(all_coord)

    if len(all_coord) == 4:
        all_coord = on_castle(all_coord, opponent_color)
    
    for i, coord in enumerate(all_coord):
        new_x, new_y = fix_coord(coord[0], opponent_color)
        all_coord[i][2] = [new_x, new_y]

    before_coord = []
    after_coord = []

    for coordinate in all_coord:
        cX, cY = coordinate[2]
        
        if determine_color(cX,cY, imageA) == opponent_color:
            before_coord = [cX,cY]
        if determine_color(cX,cY, imageB) == opponent_color:
            after_coord = [cX,cY]
    print(before_coord, after_coord)

    before_pos = get_board_coordinate(before_coord[0], before_coord[1], opponent_color)
    after_pos = get_board_coordinate(after_coord[0], after_coord[1], opponent_color)
    print(before_pos, after_pos)

    move = f'{before_pos}{after_pos}'
    return move

def opponent_made_first_move(image):
    row_x = [length*i - length//2 for i in range(1,9)]
    row_3_y = int(length*2.5)
    row_4_y = int(length*3.5)
    started = False

    for x in row_x:
        if determine_color(x, row_3_y, image) == 'white':
            started = True
        if determine_color(x,row_4_y, image) == 'white':
            started = True
    return started

def identify_opponent_first_move(image):
    row_x = [length*i - length//2 for i in range(1,9)]
    row_1_y = int(length*0.5)
    row_3_y = int(length*2.5)
    row_4_y = int(length*3.5)

    on_row_4 = False

    before_coord = []
    after_coord = []

    for x in row_x:
        if determine_color(x,row_4_y, image) == 'white':
            after_coord = [x, row_4_y]
            before_coord = [x, row_4_y - length * 2]
            on_row_4 = True
            
        if determine_color(x,row_3_y, image) == 'white':
            after_coord = [x, row_3_y]
    
    if not(on_row_4):
        for x1 in row_x:
            if determine_color(x1, row_1_y, image) is None:
                before_coord = [x1, row_1_y]
    
    before_pos = get_board_coordinate(before_coord[0], before_coord[1], 'white')
    after_pos = get_board_coordinate(after_coord[0], after_coord[1], 'white')

    move = f'{before_pos}{after_pos}'

    return move

#move mouse to the left side of the screen to start
def waiting_to_start():
    while 1:
        if ptg.position()[0] <= 20:
            break
    print('alright let\'s go')

def capture_image_now():
    image2 = np.array(ImageGrab.grab(bbox=bbox))
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

#main function (mainly logic for generating best move)
async def main():
    global length
    waiting_to_start()
    time.sleep(1)
    transport, engine = await chess.engine.popen_uci("stockfish_14_x64_avx2.exe")
    board = chess.Board()
    player_color, opponent_color = playing_color()
    
    if player_color is None:
        return
    
    if player_color == 'white':
        turn_index = 0
    else:
        turn_index = 1

    first_move = True
    
    
    while not board.is_game_over():
        if turn_index % 2 == 0:
            result = await engine.play(board, chess.engine.Limit(depth=18))
            
            ai_move = str(result.move)
            before_pos = ai_move[:2]
            after_pos = ai_move[2:]
            before_piece = board.piece_at(chess.parse_square(before_pos))

            is_pawn = True if str(before_piece).lower() == 'p' else False
            #time.sleep(randint(1,5))

            play_move(player_color, before_pos,after_pos, is_pawn)
            board.push(result.move)
            time.sleep(0.5)
        else:
            image1 = np.array(ImageGrab.grab(bbox=bbox))
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            #print(opponent_made_first_move(image1), first_move, player_color == 'black')

            if opponent_made_first_move(image1) and first_move and player_color == 'black':
                move = identify_opponent_first_move(image1)
                print(f'move: {move}')
                board.push_san(move)
            else:
                try:
                    image2 = capturing_second_image(image1)
                    cv2.imwrite('z_before.png', image1)
                    cv2.imwrite('z_after.png', image2)
                    move = determine_opponent_move(image1,image2,opponent_color)
                    print(f'move: {move}')
                    board.push_san(move)
                except:
                    image1 = capture_image_now()
                    image2 = capturing_second_image(image1)
                    cv2.imwrite('z_before_backup.png', image1)
                    cv2.imwrite('z_after_backup.png', image2)
                    move = determine_opponent_move(image1,image2,opponent_color)
                    print(f'move: {move}')
                    board.push_san(move)
            first_move = False
        turn_index += 1
    await engine.quit()
    #subprocess.call(['python', 'lichess_main.py'])

if __name__ == '__main__':
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(main())
