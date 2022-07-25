def show_demo():
    import os
    import cv2
    import pprint
    import tensorflow as tf
    import numpy as np 

    img = cv2.imread('images/dog.jpeg')
    w = int(img.shape[0])
    h = int(img.shape[1])

    def structure_print():
        print('')
        print(f'model: {os.path.basename(model_tflite)}')
        print('')
        print('==INPUT============================================')
        pprint.pprint(interpreter.get_input_details())
        print('')
        print('==OUTPUT===========================================')
        pprint.pprint(interpreter.get_output_details())

    model_tflite = 'weights/lite-model_efficientdet_lite0_detection_default_1.tflite'
    interpreter = tf.lite.Interpreter(model_tflite, num_threads=4)
    interpreter.allocate_tensors()
    structure_print()

    in_frame = cv2.resize(img, (320, 320))
    in_frame = in_frame.reshape((1, 320, 320, 3))
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, in_frame)
    interpreter.invoke()

    bboxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])
    confs = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])


    print(bboxes.shape)
    print(bboxes)
    print(class_ids.shape)
    print(class_ids) # We need to add +1 to the index of the result.
    print(confs.shape)
    print(confs)

    box = bboxes[0][0]
    cv2.rectangle(img, (int(box[1] * h), int(box[0] * w)), (int(box[3] * h), int(box[2] * w)), (0,255,0), 2, 16)

    box = bboxes[0][1]
    cv2.rectangle(img, (int(box[1] * h), int(box[0] * w)), (int(box[3] * h), int(box[2] * w)), (0,255,0), 2, 16)

    box = bboxes[0][2]
    cv2.rectangle(img, (int(box[1] * h), int(box[0] * w)), (int(box[3] * h), int(box[2] * w)), (0,255,0), 2, 16)

    # if you want to see the results on linux server, you might use the following line:
    cv2.imwrite('results/dog_result_tflite.jpeg', img) # saves demo result into results folder

    #cv2.imshow("demo", img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
