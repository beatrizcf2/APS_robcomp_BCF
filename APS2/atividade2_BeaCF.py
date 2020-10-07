import cv2
import time as t
import sys
import math
import auxiliar as aux
import numpy as np

if (sys.version_info > (3, 0)):
    # Modo Python 3
    import importlib
    importlib.reload(aux) # Para garantir que o Jupyter sempre relê seu trabalho
else:
    # Modo Python 2
    reload(aux)


#####################################################################################################################################
#funcoes úteis

def hsv_hists(img, plt):
    """
        Plota o histograma de cada um dos canais HSV
        img - imagem HSV
        plt - objeto matplotlib
    """
    plt.figure(figsize=(20,10));
    img_h = img[:,:,0]
    img_s = img[:,:,1]
    img_v = img[:,:,2]
    histo_plot(img_h, "r","H", plt);
    histo_plot(img_s, "g","S", plt);
    histo_plot(img_v, "b","V", plt);

def make_hist(img_255, c, label, plt):
    """ img_255 - uma imagem com 3 canais de 0 até 255
        c a cor do plot
        label - o label do gráfico
        plt - matplotlib.pyplot
    """
    hist,bins = np.histogram(img_255.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    # plt.plot(cdf_normalized, color = c)
    plt.hist(img_255.flatten(),256,[0,256], color = c)
    plt.xlim([0,256])
    plt.legend(label, loc = 'upper left')
    plt.plot()

def histo_plot(img, cor, label, plt):
    """
        img - imagem
        cor - cor
        plt - matplotlib.pyplot object

    """
    plt.figure(figsize=(10,5))
    make_hist(img, cor, label, plt)
    plt.show()
    plt.figure(figsize=(10,5))
    plt.imshow(img, cmap="Greys_r")#, vmin=0, vmax=255)
    plt.title(label)

def center_of_contour(contorno):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(contorno)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (int(cX), int(cY))
    
def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,5)
    cv2.line(img,(x,y - size),(x, y + size),color,5)
    
font = cv2.FONT_HERSHEY_SIMPLEX

def texto(img, a, p):
    """Escreve na img RGB dada a string a na posição definida pela tupla p"""
    cv2.putText(img, str(a), p, font,1,(0,50,100),2,cv2.LINE_AA)
    
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
####################################################################################################################################


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            input_source=int(arg) # se for um device
        except:
            input_source=str(arg) # se for nome de arquivo
    else:
        input_source = 0
        
    # iniciando a captura
    cap = cv2.VideoCapture(input_source)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            print("Codigo de retorno FALSO - problema para capturar o frame")
        
        
        # convertendo os frames para rgb e hsv
        img_bgr = frame.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        
        #Definindo os limites do inrange para as cores requeridas
        
        #ciano
        hsv1_c = np.array([90 , 220, 220], dtype=np.uint8)
        hsv2_c = np.array([100 , 255, 255], dtype=np.uint8)
        mask_c = cv2.inRange(img_hsv, hsv1_c, hsv2_c)
        #magenta
        hsv1_m = np.array([150 , 100, 200], dtype=np.uint8)
        hsv2_m = np.array([160 , 255, 255], dtype=np.uint8)
        mask_m = cv2.inRange(img_hsv, hsv1_m, hsv2_m)
        #juntando os dois em apenas uma mascara
        masks = mask_c + mask_m
        
        
        #Fazendo um blur para diminuir o ruido e transformando em binario+eliminando 'buracos'
        mask_blur = cv2.blur(masks, (3,3))
        mask = cv2.morphologyEx(mask_blur,cv2.MORPH_CLOSE,np.ones((10, 10)))
        
        
        #Definindo os contornos
        contornos, arvore = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        #Pintando os circulos e criando uma copia para podermos desenhar
        selecao = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        contornos_img = selecao.copy() # Cópia da máscara para ser desenhada "por cima"
        cv2.drawContours(contornos_img, contornos, -1, [0, 255, 0], 2);
        

        #Calculando o centro dos contornos e a distancia D
        centro = []

        for c in contornos:
            a = cv2.contourArea(c)
            if a>1000:
                p = center_of_contour(c) # centro de massa
                crosshair(contornos_img, p, 10, (128,128,0))
                centro.append(p)

        if len(contornos)>1:
            f = 2133.33 #em px
            H = 6 #centro do circulo ciano - centro circulo magenta em cm
            for i in range(1,len(centro)):
                dist = np.round(math.dist(centro[i],centro[i-1]),2)
                alfa = np.round(math.degrees(math.atan2(abs(centro[i][1]-centro[i-1][1]), abs(centro[i][0]-centro[i-1][0]))),2)
                cv2.line(contornos_img,centro[i],centro[i-1],(0,255,0),2)
                D = np.round(((f*H)/dist),2)
                cv2.putText(contornos_img, f'h = {dist}px', (10,50), font,1,(0,50,100),2,cv2.LINE_AA)
                cv2.putText(contornos_img, f'Distancia = {D}cm', (10,80), font,1,(0,50,100),2,cv2.LINE_AA)
                cv2.putText(contornos_img, f'Angulo = {alfa}graus', (10,110), font,1,(0,50,100),2,cv2.LINE_AA)
        
        
        # Hough Circles
        #Fazendo a limiarizacao para melhorar o Hough circles
        retorno, mask_limiar = cv2.threshold(mask, 100 ,255, cv2.THRESH_BINARY)
        bgr_blur = cv2.blur(img_rgb, (3,3)) #aplicando blur para melhorar detecacao de bordas
        bordas = auto_canny(bgr_blur.copy())
        circles=cv2.HoughCircles(image=mask_limiar,method=cv2.HOUGH_GRADIENT,dp=3.0,minDist=40,param1=50,param2=100,minRadius=5,maxRadius=200)
        bordas_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2RGB)

        output =  bordas_rgb.copy()


        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(output,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(output,(i[0],i[1]),2,(0,0,255),3)
                
                
        #Mostrando o resultado
        cv2.imshow('contornos', contornos_img)
        cv2.imshow('hough circles', output)

        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


