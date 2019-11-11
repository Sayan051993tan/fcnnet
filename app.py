from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from flask import Flask, request, send_file
import torchvision.transforms as T
#import numpy as np
from random import randint
import time
import io

app = Flask(__name__)

# dlab = models.segmentation.fcn_resnet101(pretrained=1).eval()

dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# dlab = torch.load('weights.pt')

# Apply the transformations needed
@app.route('/removebg', methods=['GET','POST'])
def api_root():
    if request.method == 'GET':
        return "Receiving request :)"
    elif request.method == 'POST':
        t = time.time()
        print('Working on image')
        # print(request.files["image"].read())
        img = request.files["image"]
        # segment()
        # img = Image.open(img)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

        rgb = segment(dlab, img, show_orig=False)
        plt.imshow(rgb)
        plt.axis('off')
        # plt.show()
        filename = randint(10000,99999)
        # plt.savefig('image/{}.jpg'.format(str(filename)))
        cv2.imwrite('image/{}.jpg'.format(str(filename)), rgb)
        print(time.time()- t)

        return send_file('E:/IL/removeBackground/image/{}.jpg'.format(str(filename)), 'image/jpeg', attachment_filename= '{}.jpg'.format(str(filename)))
        # return send_file(io.BytesIO(rgb),
        #                  mimetype='image/jpeg',
        #                  as_attachment=True,
        #                  attachment_filename='%s.jpg' % "image")



# Define the helper function
def decode_segmap(image, source, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255,255,255),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (255, 255, 255), (255, 255, 255), (255, 255, 255), (255,255,255), (255, 255, 255),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (255, 255, 255), (255, 255, 255), (255,255,255), (255, 255, 255), (255, 255, 255),
                             '''(255, 255, 255), (255, 255, 255), (255,255,255), (255, 255, 255), (255, 255, 255)'''])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)

    # Load the foreground input image
    foreground = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    # cv2.imshow('foreground1', foreground)
    cv2.waitKey(0)

    # Convert RGB to BGR
#     foreground = foreground[:, :, ::-1].copy() #cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)#cv2.imread(source)

    # Change the color of foreground image to RGB
    # and resize image to match shape of R-band in RGB output map
    #foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))
    # cv2.imshow('foreground2', foreground)
    cv2.waitKey(0)

    # Create a background array to hold white pixels
    # with the same size as RGB output map
    background = 255 * np.ones_like(rgb).astype(np.uint8)

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('alpha', alpha)
    cv2.waitKey(0)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) /255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # cv2.imshow('foreground', foreground)
    cv2.waitKey(0)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # cv2.imshow('background', background)
    cv2.waitKey(0)

    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)
    # cv2.imshow('outimage', outImage)
    cv2.waitKey(0)

    print(foreground.shape, image.shape)
    # outImage = cv2.bitwise_and(foreground,image)
    # cv2.imshow('outimage', outImage)
    # cv2.waitKey(0)


    # Return a normalized output image for display
    return outImage


def segment(net, path, show_orig=True, dev='cpu'):

    img = Image.open(path)
    print(img.size)

    if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([T.Resize(450),
                     #T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    #path = path.read()
    # plt.imshow(om)
    # plt.axis('off')
    # plt.show()
    rgb = decode_segmap(om, img)


    return rgb


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)

