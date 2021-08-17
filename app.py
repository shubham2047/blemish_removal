import cv2
import numpy as np

img_path = './imgs/blemish.jpg'
img_save_path = './imgs/output.jpg'
img =  cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
h, w = img.shape[:2]

r = min(int(.01*h), int(.01*w))

max_r = min(int(.04*h), int(.04*w))

print(img.shape)
img_copy = img.copy()
img_old = img.copy()

def getWindowSize(img):
        h_ratio = 960 / img.shape[0] 
        w_ratio = 640 / img.shape[1] 
        interpolation = cv2.INTER_LINEAR
        return cv2.resize(img, tuple([0,0]), fx=max(h_ratio, w_ratio), fy=max(h_ratio, w_ratio), interpolation=interpolation).shape[:2]

def radiusSelector(*args):
    global r
    r = args[0]

def callbackFunction(events,x,y,flags,param):
    global pt
    if events == cv2.EVENT_LBUTTONDOWN:
        pt = (x,y)
        removeBlemish(x,y)

def removeBlemish(x,y):
    global img_old,img_copy,r
    row = y
    col = x
    a,b,c,d = row-r,row+r+1,col-r,col+r+1
    if (a<0) or (c<0) or (b>img.shape[0]) or (d>img.shape[1]):
        return
    sobel_x = cv2.Sobel(img_copy[a:b,c:d],cv2.CV_32F,1,0,ksize=-1)
    sobel_y = cv2.Sobel(img_copy[a:b,c:d],cv2.CV_32F,0,1,ksize=-1)
    best_gradient_mean = np.mean(np.sqrt((np.square(sobel_x)+np.square(sobel_y))))
    best_window = [a,b,c,d]
    u_row,d_row,l_col,r_col = row-2*r-1,row+2*r+1,col-2*r-1,col+2*r+1
    neighbor_list = [(row-r,row+r+1,l_col-r,l_col+r+1),
                     (u_row-r,u_row+r+1,col-r,col+r+1),
                     (row-r,row+r+1,r_col-r,r_col+r+1),
                     (d_row-r,d_row+r+1,col-r,col+r+1)]
    for a,b,c,d in neighbor_list:
        if (a>=0) and (c>=0) and (b<=img.shape[0]) and (d<=img.shape[1]) :
            sobel_x = cv2.Sobel(img_copy[a:b,c:d],cv2.CV_32F,1,0,ksize=-1)
            sobel_y = cv2.Sobel(img_copy[a:b,c:d],cv2.CV_32F,0,1,ksize=-1)
            gradient_mean = np.mean(np.sqrt((np.square(sobel_x)+np.square(sobel_y))))
            if gradient_mean < best_gradient_mean:
                best_gradient_mean = gradient_mean
                best_window = [a,b,c,d]
    if best_window != [row-r,row+r+1,col-r,col+r+1] :
        img_old = img_copy
        mask = np.ones(img_copy[best_window[0]:best_window[1],best_window[2]:best_window[3]].shape,dtype = np.uint8)
        mask = mask*255
        img_copy = cv2.seamlessClone(img_copy[best_window[0]:best_window[1],best_window[2]:best_window[3]],img_copy,mask,(x,y),cv2.NORMAL_CLONE)
        cv2.imshow('window',img_copy)

win_h, win_w = getWindowSize(img)

cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('window', win_w, win_h)

cv2.createTrackbar('Blemish radius :', 'window', r, max_r, radiusSelector)

cv2.imshow('window',img)


print()
print('################## Blemish Removal ##################')
print()
print('Click on blemish which you want to remove.')
print('Change "Blemish radius" size if you are not getting good result.')
print('press "z" to undo......')
print('press "s" to save......')
print('press "esc" to quit')

cv2.setMouseCallback('window',callbackFunction)

while True:
    
    k = cv2.waitKey(0)
    if k == 27:
        break
    if k == ord('z'):
        img_copy = img_old
        cv2.imshow('window',img_copy)
    if k== ord('s'):
        cv2.imwrite(img_save_path, img_copy)
        print("image saved!")
cv2.destroyAllWindows()