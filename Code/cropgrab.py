import cv2
import numpy as np
class CropGrab:
    def add_contours(self, image, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0) ,2)

        return x,y,w,h

    def remove_background(self, image,image1,n,p,img_name, loop):
        h, w = image.shape[:2]
        mask = self.init_grabcut_mask(h, w, loop)
        bgm = np.zeros((1, 65), np.float64)
        fgm = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
        mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = cv2.bitwise_and(image, image1, mask = mask_binary)
        x,y,w,h = self.add_contours(result, mask_binary) # optional, adds visualizations
        result1 = image[y:y+h, x:x+w]
        cv2.imwrite(img_name,result1)

    def init_grabcut_mask(self, h, w, s):
        if s == 1:
            mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
            mask[:14*h//15, :w//10] = cv2.GC_PR_FGD
            mask[:18*h//20, w//10:2*w//10] = cv2.GC_PR_FGD
            mask[2*h//16:14*h//16, w//6:13*w//48] = cv2.GC_PR_FGD
            mask[3*h//20:17*h//20, 13*w//48:15*w//48] = cv2.GC_PR_FGD
            mask[7*h//40:25*h//30, 15*w//48:17*w//48] = cv2.GC_PR_FGD
            mask[9*h//40:24*h//30, 17*w//48:19*w//48] = cv2.GC_PR_FGD
            mask[11*h//40:23*h//30, 19*w//48:21*w//48] = cv2.GC_PR_FGD
            mask[13*h//40:21*h//30, 21*w//48:22*w//48] = cv2.GC_PR_FGD
            mask[15*h//40:20*h//30, 22*w//48:23*w//48] = cv2.GC_PR_FGD

            mask[h//24:16*h//18, :7*w//80] = cv2.GC_FGD
            mask[h//14:12*h//14, 7*w//80:3*w//20] = cv2.GC_FGD
            mask[2*h//12:10*h//12, 3*w//20:5*w//20] = cv2.GC_FGD
            mask[4*h//20:16*h//20, 5*w//20:13*w//40] = cv2.GC_FGD
            mask[5*h//20:23*h//30, 13*w//40:15*w//40] = cv2.GC_FGD
            mask[9*h//30:22*h//30, 15*w//40:16*w//40] = cv2.GC_FGD
            mask[10*h//30:21*h//30, 16*w//40:17*w//40] = cv2.GC_FGD
            mask[11*h//30:20*h//30, 17*w//40:18*w//40] = cv2.GC_FGD
            mask[12*h//30:19*h//30, 18*w//40:37*w//80] = cv2.GC_FGD
        elif s == 2:
            mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
            mask[h//15:14*h//15, :w//10] = cv2.GC_PR_FGD
            mask[2*h//20:18*h//20, w//10:2*w//10] = cv2.GC_PR_FGD
            mask[2*h//16:14*h//16, w//6:13*w//48] = cv2.GC_PR_FGD
            mask[3*h//20:17*h//20, 13*w//48:15*w//48] = cv2.GC_PR_FGD
            mask[7*h//40:25*h//30, 15*w//48:17*w//48] = cv2.GC_PR_FGD
            mask[9*h//40:24*h//30, 17*w//48:19*w//48] = cv2.GC_PR_FGD
            mask[11*h//40:23*h//30, 19*w//48:21*w//48] = cv2.GC_PR_FGD
            mask[13*h//40:21*h//30, 21*w//48:22*w//48] = cv2.GC_PR_FGD
            mask[15*h//40:20*h//30, 22*w//48:23*w//48] = cv2.GC_PR_FGD

            mask[2*h//18:16*h//18, :7*w//80] = cv2.GC_FGD
            mask[2*h//14:12*h//14, 7*w//80:3*w//20] = cv2.GC_FGD
            mask[2*h//12:10*h//12, 3*w//20:5*w//20] = cv2.GC_FGD
            mask[4*h//20:16*h//20, 5*w//20:13*w//40] = cv2.GC_FGD
            mask[5*h//20:23*h//30, 13*w//40:15*w//40] = cv2.GC_FGD
            mask[9*h//30:22*h//30, 15*w//40:16*w//40] = cv2.GC_FGD
            mask[10*h//30:21*h//30, 16*w//40:17*w//40] = cv2.GC_FGD
            mask[11*h//30:20*h//30, 17*w//40:18*w//40] = cv2.GC_FGD
            mask[12*h//30:19*h//30, 18*w//40:37*w//80] = cv2.GC_FGD
        return mask 