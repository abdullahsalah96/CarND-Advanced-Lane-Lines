import numpy as np
WIDTH = 1280            #width of the resized frame
HEIGHT = 720            #height of resized frame
ym_per_pix = 30 / 720   # meters per pixel in y dimension - should be changed according to setup of each camera
xm_per_pix = 5 / 1280  # meters per pixel in x dimension - shoudl be changed according to setup of each camera
time_window = 10        # results are averaged over this number of frames
FLIP = False            # set true when using camera as video feed is flipped
L_CHANNEL_MIN = 160     #min L_Channel used to tune L channel thresholding as it sometimes changes from different videos
SOBEL_MIN_THRESH = 1   #min Sobel threshold used to tune sobel threshold as it sometimes changes from different videos

src = np.float32([[WIDTH, HEIGHT-10],    # bottom right
                    [0, HEIGHT-10],       # bottom left
                    [546, 460],           # top left
                    [732, 460]])          # top right

dst = np.float32([[WIDTH, HEIGHT],        # bottom right
                    [0, HEIGHT],          # bottom left
                    [0, 0],               # top left
                    [WIDTH, 0]])          # top right