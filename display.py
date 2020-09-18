from pygame import locals
import pygame as py
import numpy as np
import sys
import os
import ui

py.init()

create_training_data = False

buttons = []
input_field = ui.input_field

pixel_width = 20
pixel_amount = 25

size = (0,0)
window = py.surface

def Transfer_Data(li):
    global input_field, create_training_data

    color_list = []
    i = 0
    while i < len(li):
        color_list.append(li[i].color)
        i += 1
    
    if create_training_data:
        try:
            x_list = np.load(os.path.join(sys.path[0], 'data', 'x_list.npy'))
            np.save(os.path.join(sys.path[0], 'data', 'x_list.npy'),np.vstack((x_list, np.array(color_list))))
        except:
            np.save(os.path.join(sys.path[0], 'data', 'x_list.npy'),np.array(color_list))
        
        try:
            y_list = np.load(os.path.join(sys.path[0], 'data', 'y_list.npy'))
            np.save(os.path.join(sys.path[0], 'data', 'y_list.npy'),np.vstack((y_list, np.array([int(input_field.text)]))))
        except Exception as e:
            np.save(os.path.join(sys.path[0], 'data', 'y_list.npy'), np.array(int(input_field.text)))
            print(e)
        
        try:
            descrip_binary = np.load(os.path.join(sys.path[0], 'data', "data_description_binary.npy"))
            descrip_binary[int(input_field.text)] = descrip_binary[int(input_field.text)] + 1
            np.save(os.path.join(sys.path[0], 'data', 'data_description_binary.npy'), descrip_binary)
            string = f"0: {descrip_binary[0]}\t1: {descrip_binary[1]}\n2: {descrip_binary[2]}\t3: {descrip_binary[3]}\n4: {descrip_binary[4]}\t5: {descrip_binary[5]}\n6: {descrip_binary[6]}\t7: {descrip_binary[7]}\n8: {descrip_binary[8]}\t9: {descrip_binary[9]}\n"
            descrip_read = open(os.path.join(sys.path[0], 'data', "data_description_readable.txt"), "w")
            descrip_read.write(string)
            descrip_read.close()
        except:
            descrip_binary =  np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
            descrip_binary[int(input_field.text)] = descrip_binary[int(input_field.text)] + 1
            np.save(os.path.join(sys.path[0], 'data', 'data_description_binary.npy'), descrip_binary)
            descrip_read = open(os.path.join(sys.path[0], 'data', "data_description_readable.txt"), "w")
            string = f"0: {descrip_binary[0]}\t1: {descrip_binary[1]}\n2: {descrip_binary[2]}\t3: {descrip_binary[3]}\n4: {descrip_binary[4]}\t5: {descrip_binary[5]}\n6: {descrip_binary[6]}\t7: {descrip_binary[7]}\n8: {descrip_binary[8]}\t9: {descrip_binary[9]}\n"
            descrip_read.write(string)
            descrip_read.close()

    else:
        np.save(os.path.join(sys.path[0], 'data', "user_input.npy"), np.array(color_list))
    
    #raise SystemExit()

def Process_Data(li):
    global pixel_amount
    list_2d_raw = []
    i = 1
    j = 0
    while i < pixel_amount + 1:
        list_2d_raw.append([])
        list_2d_raw[i - 1] = li[j: pixel_amount * i]
        j = pixel_amount * i
        i += 1

    list_2d_processed = []
    i = 0
    while i < pixel_amount:
        list_2d_processed.append([])
        j = 0
        while j < pixel_amount:
            list_2d_processed[i].append(list_2d_raw[i][j].color)
            j += 1
        i += 1

    i = 0
    while i < pixel_amount:
        j = 0
        while j < pixel_amount:
            value = 0.0
            list_2d_processed[i][j] = 0
            if i > 0 and i < pixel_amount - 1:
                if j > 0 and j < pixel_amount - 1:
                    value += list_2d_raw[i-1][j-1].color
                    value += list_2d_raw[i-1][j].color
                    value += list_2d_raw[i-1][j+1].color
                    value += list_2d_raw[i][j-1].color
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i][j+1].color
                    value += list_2d_raw[i+1][j-1].color
                    value += list_2d_raw[i+1][j].color
                    value += list_2d_raw[i+1][j+1].color
                    list_2d_processed[i][j] = value/9
                elif not j > 0:
                    value += list_2d_raw[i-1][j].color
                    value += list_2d_raw[i-1][j+1].color
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i][j+1].color
                    value += list_2d_raw[i+1][j].color
                    value += list_2d_raw[i+1][j+1].color
                    list_2d_processed[i][j] = value/6
                else:
                    value += list_2d_raw[i-1][j-1].color
                    value += list_2d_raw[i-1][j].color
                    value += list_2d_raw[i][j-1].color
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i+1][j-1].color
                    value += list_2d_raw[i+1][j].color
                    list_2d_processed[i][j] = value/6
            elif not i > 0:
                if j > 0 and j < pixel_amount - 1:
                    value += list_2d_raw[i][j-1].color
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i][j+1].color
                    value += list_2d_raw[i+1][j-1].color
                    value += list_2d_raw[i+1][j].color
                    value += list_2d_raw[i+1][j+1].color
                    list_2d_processed[i][j] = value/6
                elif not j > 0:
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i][j+1].color
                    value += list_2d_raw[i+1][j].color
                    value += list_2d_raw[i+1][j+1].color
                    list_2d_processed[i][j] = value/4
                else:
                    value += list_2d_raw[i][j-1].color
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i+1][j-1].color
                    value += list_2d_raw[i+1][j].color
                    list_2d_processed[i][j] = value/4
            else:
                if j > 0 and j < pixel_amount - 1:
                    value += list_2d_raw[i-1][j-1].color
                    value += list_2d_raw[i-1][j].color
                    value += list_2d_raw[i-1][j+1].color
                    value += list_2d_raw[i][j-1].color
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i][j+1].color
                    list_2d_processed[i][j] = value/6
                elif not j > 0:
                    value += list_2d_raw[i-1][j].color
                    value += list_2d_raw[i-1][j+1].color
                    value += list_2d_raw[i][j].color
                    value += list_2d_raw[i][j+1].color
                    list_2d_processed[i][j] = value/4
                else:
                    value += list_2d_raw[i-1][j-1].color
                    value += list_2d_raw[i-1][j].color
                    value += list_2d_raw[i][j-1].color
                    value += list_2d_raw[i][j].color
                    list_2d_processed[i][j] = value/4
            j += 1
            #print(f"v{value}  i{i}  j{j}")
        i += 1
    
    i = 0
    while i < pixel_amount:
        j = 0
        while j < pixel_amount:
            #print(list_2d_raw[i][j].color)
            list_2d_raw[i][j].change_color(1.5 * list_2d_processed[i][j])
            j += 1
        i += 1

def Initialise(create, pw, pa):
    global pixel_amount, pixel_width, size, window, input_field, create_training_data, buttons

    buttons.clear()
    create_training_data = create
    pixel_width = pw
    pixel_amount = pa

    width = ((pixel_width + round(pixel_width / 8)) * pixel_amount) + pixel_width
    height = width + 50 
    size = width,height
    window = py.display.set_mode(size)
    window.fill((65, 85, 105))

    i = 0
    while i < pixel_amount:
        j = 0
        while j < pixel_amount:
            row_pos = (pixel_width + round(pixel_width/8)) * j + round(pixel_width/2)
            col_pos = (pixel_width + round(pixel_width/8)) * i + round(pixel_width/2)
            buttons.append(ui.button(col_pos, row_pos, pixel_width, pixel_width, window))
            py.display.update()
            j += 1
        i += 1
    
    if_pos_x = buttons[0].pos_x
    if_pos_y = size[1]-50
    if_width = (buttons[len(buttons) - 1].pos_x - buttons[0].pos_x) + pixel_width
    if_height = 40
    input_field = ui.input_field(if_pos_x, if_pos_y, if_width, if_height, window)

def Run():
    global input_field, window
    up = True
    while True:
        left_click , scroll_click, right_click = py.mouse.get_pressed()
        if left_click == 1:
            up = False
            for b in buttons:
                b.clicked(py.mouse.get_pos())
            input_field.clicked(py.mouse.get_pos())
            py.display.flip()

        if right_click == 1 and up:
            up = False
            Process_Data(buttons)
            Transfer_Data(buttons)
            return

        for event in py.event.get():
            if event.type == py.MOUSEBUTTONUP:
                up = True

            if event.type == py.QUIT:
                raise SystemExit()

'''
while True:
    Initialise(True, 20, 25)
    Run()
'''