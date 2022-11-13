import cv2 
from DBf import *
from PIL import Image, ImageTk, ImageDraw, ImageFont




# =======================================================================================================
# Detection Functions
# =======================================================================================================
def detect_one_color(rgb_arr, bmin, bmax):
    len_arr_w = rgb_arr.shape[0]
    len_arr_h = rgb_arr.shape[1]
    len_arr = len_arr_w * len_arr_h

    # print(f'{len_arr_w = }')
    # print(f'{len_arr_h = }')
    # print(f'{len_arr = }')

    n = 0
    for item_row in rgb_arr:
        for item_column in item_row:
            # print(f'{item_column = }')
            if bmin < item_column < bmax:
                n += 1
    return n / len_arr 


# =======================================================================================================
# Color Recognition and Comparison Function
# =======================================================================================================
def color_detect():

    face_recog_frame = 0
    # video_path = 1
    # # video_path = 'rtsp://url'
    # cap = cv2.VideoCapture(video_path)
    

    # while True:
    if True:

        # face_recog_frame += 1
        # ref, frame = cap.read()
        faces_key_points = []
        faces_box_points = []
        # cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pilImage = Image.fromarray(cvimage)

        
        a = 14

        img_name = str(a) + '.png'
        pilImage = Image.open('./testimgs/' + img_name)


        dr_pilImage = ImageDraw.Draw(pilImage)

        frame = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)

        if face_recog_frame % 1 == 0:
            dbface = DBFace()
            dbface.eval()

            if HAS_CUDA:
                dbface.cuda()

            dbface.load("./model/dbface.pth")
            objs = detect(dbface, frame)

            faces_key_points.clear()

            for obj in objs:
                # print(f'\n{obj = }')
                # print(f'{obj.landmark = }')
                # print(f'{type(obj) = }')
                faces_key_points.append(obj.landmark)
                faces_box_points.append([obj.x, obj.y, obj.width, obj.height])

            # obj = (BBox[0]: x=258.63, y=248.73, r=362.24, b=383.68, width=104.61, height=135.95, landmark=(269.6686782836914, 305.2546682357788),(319.151083946228, 294.79078674316406),(293.52794456481934, 321.12305307388306),(285.3873538970947, 349.17039489746094),(324.22133445739746, 340.1794891357422))
            # obj.landmark = [(244.0751495361328, 185.55762672424316), (305.96776390075684, 185.87062454223633), (268.0561294555664, 209.2957124710083), (249.75679397583008, 245.1006202697754), (299.11580181121826, 247.66641235351562)]
            # ========================================================================================================
            #  Face boxes
            # len_faces_key_points = len(faces_key_points)




            # Save pure protogenetic pictures
            # ========================================================================================================
            # Charaterisitcs Detection and Recognition
            img_save_path = './img/'
            i = 0
            for tuple_i in faces_box_points:
                item = tuple_i
                # print(f'{item = }')
                x1 = int(item[0])
                y1 = int(item[1])
                x2 = int((item[0] + item[2]))
                y2 = int((item[1] + item[3]))

                # ________________________________________________________________________
                i += 1
                img = pilImage.crop((x1, y1, x2, y2))
                img = img.convert("RGB")
                img.save(img_save_path + str(i) + '.jpg')
                # ________________________________________________________________________
                dr_pilImage.rectangle((x1, y1, x2, y2), fill=None, outline=(0, 255, 0), width=3)
            # ========================================================================================================



            # Mask Detection
            # ========================================================================================================
            r = 5
            for tuple_i in range(len(faces_key_points)):
                tuple_item = faces_key_points[tuple_i]
                box_xywh = faces_box_points[tuple_i]

                left_eye = tuple_item[0]
                right_eye = tuple_item[1]
                nose = tuple_item[2]
                left_lip = tuple_item[3]
                right_lip = tuple_item[4]

                eye_central_x = int((left_eye[0] + right_eye[0]) / 2)
                eye_central_y = int((left_eye[1] + right_eye[1]) / 2)
                lip_central_x = int((left_lip[0] + right_lip[0]) / 2)
                lip_central_y = int((left_lip[1] + right_lip[1]) / 2)

                box_w = int(0.1 * box_xywh[2])
                box_h = int(0.2 * box_xywh[3])

                adjust = 0.5
                box_1_x1 = eye_central_x - box_w
                box_1_y1 = eye_central_y - box_h * adjust
                box_1_x2 = eye_central_x + box_w
                box_1_y2 = eye_central_y

                box_2_x1 = lip_central_x - box_w
                box_2_y1 = lip_central_y - box_h
                box_2_x2 = lip_central_x + box_w
                box_2_y2 = lip_central_y + box_h

                eye_nose_box_coordinates = (box_1_x1, box_1_y1, box_1_x2, box_1_y2)
                nose_lip_box_coordinates = (box_2_x1, box_2_y1, box_2_x2, box_2_y2)

                if (box_1_x1 < box_1_x2) and (box_1_y1 < box_1_y2) and \
                    (box_2_x1 < box_2_x2) and (box_2_y1 < box_2_y2):
                    eye_nose_box = pilImage.crop(eye_nose_box_coordinates)
                    nose_lip_box = pilImage.crop(nose_lip_box_coordinates)


                    box1_r_g_b_image = eye_nose_box.split()
                    red_eye_nose_box = np.asarray(box1_r_g_b_image[0])
                    green_eye_nose_box = np.asarray(box1_r_g_b_image[1])
                    blue_eye_nose_box = np.asarray(box1_r_g_b_image[2])


                    box2_r_g_b_image = nose_lip_box.split()
                    red_nose_lip_box = np.asarray(box2_r_g_b_image[0])
                    green_nose_lip_box = np.asarray(box2_r_g_b_image[1])
                    blue_nose_lip_box = np.asarray(box2_r_g_b_image[2])


                    color_threshold = 100
                    box1_red_acc = detect_one_color(red_eye_nose_box, color_threshold, 255)
                    box1_green_acc = detect_one_color(green_eye_nose_box, color_threshold, 255)
                    box1_blue_acc = detect_one_color(blue_eye_nose_box, color_threshold, 255)


                    color_threshold = 100
                    box2_red_acc = detect_one_color(red_nose_lip_box, color_threshold, 255)
                    box2_green_acc = detect_one_color(green_nose_lip_box, color_threshold, 255)
                    box2_blue_acc = detect_one_color(blue_nose_lip_box, color_threshold, 255)


                    print()
                    print(f'{box1_blue_acc = }')
                    print(f'{box1_red_acc = }')
                    print(f'{box1_green_acc = }')

                    print()
                    print(f'{box2_blue_acc = }')
                    print(f'{box2_red_acc = }')
                    print(f'{box2_green_acc = }')


                    HasMask = False
                    if box2_blue_acc > 0.8 and box2_red_acc > 0.8 and box2_green_acc > 0.8 and \
                        box1_blue_acc < 0.9:
                        HasMask = True
                    if box2_blue_acc > 0.8 and box2_red_acc > 0.8 and box2_green_acc < 0.2:
                        HasMask = True
                    if box2_blue_acc > 0.8 and box2_red_acc < 0.2 and box2_green_acc > 0.8:
                        HasMask = True
                    if box2_blue_acc > 0.8 and box2_red_acc < 0.2 and box2_green_acc < 0.2:
                        HasMask = True
                    if box2_blue_acc < 0.2 and box2_red_acc > 0.8 and box2_green_acc < 0.2:
                        HasMask = True
                    if box2_blue_acc < 0.2 and box2_red_acc < 0.2 and box2_green_acc > 0.8:
                        HasMask = True
                    if box1_blue_acc > 0.4 and box1_red_acc > 0.9 and box1_green_acc > 0.8 and \
                        box2_blue_acc > 0.4 and box2_red_acc < 0.1 and box2_green_acc > 0.4:
                        HasMask = True
                    if 0.2 < box2_blue_acc < 0.5 and box2_red_acc < 0.2 and box2_green_acc < 0.2:
                        HasMask = True
                    if box2_blue_acc == 0 and box2_red_acc == 0 and box2_green_acc == 0:
                        HasMask = True


                    print(f'{HasMask = }')

                    dr_pilImage.rectangle(eye_nose_box_coordinates, fill=None, outline=(0, 255, 0), width=2)

                    if HasMask:
                        dr_pilImage.rectangle(nose_lip_box_coordinates, fill=None, outline=(0, 255, 255), width=2)
                    else:
                        dr_pilImage.rectangle(nose_lip_box_coordinates, fill=None, outline=(255, 255, 0), width=2)


                color = (255, 99, 0)
                for i in range(5):
                    item = tuple_item[i]
                    x = int(item[0])
                    y = int(item[1])
                    dr_pilImage.ellipse((x, y, x+r, y+r), fill=color)
            # ========================================================================================================


        pilImage.show()
        pilImage.save('./testsave/' + img_name)


        # cv2.namedWindow("Accurate and Fast Mask Recognition", cv2.WINDOW_NORMAL) 
        # cv2.resizeWindow("Accurate and Fast Mask Recognition", 1000, 666)
        # image = np.array(pilImage)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Accurate and Fast Mask Recognition', image)
        # cv2.waitKey(1)




# ========================================================================================================
# Main Function
if __name__ == "__main__":
    color_detect()
