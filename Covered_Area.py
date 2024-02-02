import cv2
import numpy as np
from scipy.special import gamma
from scipy.stats import kstest
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import sys
import pandas as pd
import datetime
import re


class MainClass(object):

    def __init__(self):

        self.img_o = np.zeros((1, 1, 3), dtype='float32')
        self.img_find_lig = np.zeros((1, 1, 3), dtype='float32') #imagem com fundo BRANCO para contabilizar os pixels pretos do ligante
        self.img_find_bri = np.zeros((1, 1, 3), dtype='float32') #imagem com fundo PRETO para contabilizar os pixels pretos do brilho do ligante
        self.img_find_agr = np.zeros((1, 1, 3), dtype='float32') #imagem com fundo PRETO e partículas BRANCAS para contar os agregados (máscara completa)
        self.img_find_mrc = np.zeros((1, 1, 3), dtype='float32') #imagem com partículas marcadas de vermelho para avaliar a limiarização
        
        self.img_count_lig = np.zeros((1, 1, 3), dtype='float32') #imagem com pixels de ligante BRANCOS após limiarização (para contagagem)
        self.img_count_bri = np.zeros((1, 1, 3), dtype='float32') #imagem com pixels de brilho BRANCOS após limiarização (para contagagem)

        self.i_ligante = np.zeros((1, 1, 3), dtype='float32') #imagem com pixels de ligante VERMELHOS após limiarização
        self.i_brilho = np.zeros((1, 1, 3), dtype='float32') #imagem com pixels de brilho VERMELHOS após limiarização

        self.id = []
        self.nAgreg = []
        self.nLigan = []
        self.nBrilh = []
        self.nCobrimento = []
        self.estatistic = []
        self.valor_limiar_lig = 0
        self.valor_limiar_bri = 0

        self.pasta = 'C:/Covered_Area_results'
        fontsize_base = 12
        fonte1 = ("Perpetua", fontsize_base)
        fonte2 = ("Perpetua", fontsize_base-2)
        fonte3 = ("Perpetua", fontsize_base, "bold")
        fonte4 = ("Perpetua", fontsize_base+4, "bold")

        self.list_contours = []

        # self.absolute_path = os.path.dirname(__file__)
        self.absolute_path = 'C:/Covered_Area_data/A12 - Maranhao/imagem_escolhida'

        self.root = Tk()
        self.root.title("Covered Area")
        self.root.state('zoomed')
        self.root.resizable(width=True, height=False)

        self.fAcao = LabelFrame(self.root, text="Ações", font=fonte1)
        self.fLimiares = LabelFrame(self.root, text="Controle de limiares", font=fonte1)
        self.fGraphics = LabelFrame(self.root, text="Estatística", font=fonte1, width=500)
        self.fImagens = LabelFrame(self.root, text="Exibição de imagens", font=fonte1)

        self.fAcao.grid(row=0, column= 0, columnspan= 2, padx=5, sticky='WE')
        self.fLimiares.grid(row=1, column= 0, padx=5)
        self.fGraphics.grid(row=1, column= 1, padx=5, sticky='NS')
        self.fImagens.grid(row=2, column= 0, columnspan= 2, padx=5, sticky='WE')

        self.lOriginal = Label(self.fImagens, text="Imagem original", font=fonte4)
        self.lObjetos = Label(self.fImagens, text="Objetos detectados", font=fonte4)
        self.lLigante = Label(self.fImagens, text="Realce do Ligante", font=fonte4)
        self.lBrilho = Label(self.fImagens, text="Realce do Brilho do ligante", font=fonte4)

        self.lOriginal.grid(row=0, column= 0)
        self.lObjetos.grid(row=0, column= 1)
        self.lLigante.grid(row=0, column= 2)
        self.lBrilho.grid(row=0, column= 3)
        
        self.bAbrir = Button(self.fAcao, text="Abrir imagem", font=fonte3, width= 10, cursor="hand2", command=self.openImg)
        self.bReiniciar = Button(self.fAcao, text="Reiniciar", font=fonte3, fg="red", width= 10, cursor="hand2", command=self.reiniciar)
        self.bProcessar = Button(self.fAcao, text="Processar", font=fonte3, width= 10, cursor="hand2", command=self.processar)
        self.bCalcular = Button(self.fAcao, text="Calcular", font=fonte3, width= 10, cursor="hand2", command=self.calcular)
        self.bGravar = Button(self.fAcao, text="Gravar", font=fonte3, width= 10, cursor="hand2", command=self.gravar)
        self.lQuantidade = Label(self.fAcao, text="Partículas: 0", font=fonte3)
        self.lCalculos = Label(self.fAcao, text="Quantidade de pixels\nAgregado: | Ligante: | Brilho: ", font=fonte2)
        self.lLocal = Label(self.fAcao, text=f"Arquivos em: {self.pasta}", font=fonte2)
        self.lPasta = Label(self.fAcao, text="Gravar na pasta:", font=fonte2)
        self.ePasta = Entry(self.fAcao, width=30, font=fonte2)

        self.bReiniciar.grid(row=0, column= 0, rowspan= 2, padx= 5, pady= 5)
        self.bAbrir.grid(row=0, column= 1, rowspan= 2, padx= 5, pady= 5)
        self.bProcessar.grid(row=0, column= 2, rowspan= 2, padx= 5, pady= 5)
        self.bCalcular.grid(row=0, column= 3, rowspan= 2, padx= 5, pady= 5)
        self.bGravar.grid(row=0, column= 4, rowspan= 2, padx= 5, pady= 5)
        self.lQuantidade.grid(row=0, column= 5, rowspan= 2, padx= 5, pady= 5)
        self.lCalculos.grid(row=0, column= 6, rowspan= 2, padx= 5, pady= 5)
        self.lLocal.grid(row=0, column= 7, columnspan= 2, padx= 5, pady= 5)
        self.lPasta.grid(row=1, column= 7, padx= 5, pady= 5)
        self.ePasta.grid(row=1, column= 8, padx= 5, pady= 5, sticky='WE')

        self.limiarTextL = Label(self.fLimiares, text="Limiar para identificar o ligante", font=fonte3)
        self.limiarTextB = Label(self.fLimiares, text="Limiar para identificar o brilho ligante", font=fonte3)
        self.limiarL = Scale(self.fLimiares, width= 20, length= 300, from_= 0, to= 255, font=fonte2, orient= HORIZONTAL, cursor="hand2", command=self.sliderL)
        self.limiarB = Scale(self.fLimiares, width= 20, length= 300, from_= 0, to= 255, font=fonte2, orient= HORIZONTAL, cursor="hand2", command=self.sliderB)
        self.cHist = Canvas(self.fLimiares, bg="black",width=256, height=256)
        self.lLimiar = Label(self.fLimiares, text="Aplicar segmentação", font=fonte3)
        self.bLimiar = Button(self.fLimiares, text="Limiares", font=fonte3, width= 10, cursor="hand2", command= lambda: self.aplicando_limiar(self.img_find_lig.copy(), self.img_find_bri.copy()))
        self.bOtsu = Button(self.fLimiares, text="Otsu", font=fonte3, width= 10, cursor="hand2", command= lambda: self.aplicando_otsu(self.img_find_lig.copy(), self.img_find_bri.copy()))

        self.cHist.grid(row= 0, column= 0, rowspan= 7)
        self.limiarTextL.grid(row= 0, column= 1, columnspan= 2)
        self.limiarL.grid(row= 1, column= 1, columnspan= 2)
        self.limiarTextB.grid(row= 2, column= 1, columnspan= 2)
        self.limiarB.grid(row= 3, column= 1, columnspan= 2)
        self.lLimiar.grid(row= 4, column= 1, columnspan= 2)
        self.bLimiar.grid(row= 5, column= 1)
        self.bOtsu.grid(row= 5, column= 2)


        self.limiarB.set(255)
        self.bProcessar.config(state="disabled")
        self.bCalcular.config(state="disabled")
        self.bGravar.config(state="disabled")
        self.bLimiar.config(state="disabled")
        self.bOtsu.config(state="disabled")
        self.limiarL.config(state="disabled")
        self.limiarB.config(state="disabled")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.fImagens.grid_columnconfigure(0, weight=1)
        self.fImagens.grid_columnconfigure(1, weight=1)
        self.fImagens.grid_columnconfigure(2, weight=1)
        self.fImagens.grid_columnconfigure(3, weight=1)

        self.fAcao.grid_columnconfigure(6, weight=1)
        self.fAcao.grid_columnconfigure(8, weight=1)
        
        self.root.mainloop()


    def reiniciar(self):
        os.execl(sys.executable, sys.executable, *sys.argv) #comando para reiniciar a aplicação

    def openImg(self):
        try:
            self.root.filename = filedialog.askopenfilename(initialdir=self.absolute_path, title="Selecione um arquivo", filetypes=(("Arquivo jpg", "*.jpg"), ("Arquivo png", "*.png")))
        except:
            pass

        if(len(str(self.root.filename)) > 0):
            i = cv2.imread(str(self.root.filename), cv2.IMREAD_UNCHANGED)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            self.img_o = i.copy()

            dir_path = os.path.dirname(str(self.root.filename))
            self.ePasta.delete(0, END)
            self.ePasta.insert(0, dir_path)

            self.lLocal.config(text="Arquio: "+str(self.root.filename))
            self.bProcessar.config(state="normal")
            self.bCalcular.config(state="disabled")
            self.bGravar.config(state="disabled")
            self.bLimiar.config(state="disabled")
            self.bOtsu.config(state="disabled")
            self.limiarL.config(state="disabled")
            self.limiarB.config(state="disabled")
            self.showImg(i, self.lOriginal)
           
    def showImg(self, img, place):
        i = img.copy()
        i = (i.astype('float32'))/255
        i_resize = self.resize(i)
        e_array = np.uint8((i_resize * 255))
        i_p = Image.fromarray(e_array)
        i_t = ImageTk.PhotoImage(i_p)
        place.config(image=i_t)
        place.image = i_t

    def resize(self, img):
        width_img = int(img.shape[1])
        height_img = int(img.shape[0])
        p=1.0

        w_r = self.root.winfo_width()
        h_r = self.root.winfo_height()
        h_fAcao = self.fAcao.winfo_reqheight()
        h_fLim = self.fLimiares.winfo_reqheight()

        max_height = h_r - (h_fAcao + h_fLim) - 27
        max_width = (w_r/4) - 5

        scale_h = max_height/height_img
        scale_w = max_width/width_img

        if scale_w < scale_h:
            p = scale_w
        else:
            p = scale_h

        width_r = int(img.shape[1]*p)
        height_r = int(img.shape[0]*p)
        dim = (width_r, height_r)
        img_r = cv2.resize(img.copy(), dim)
        return img_r

    def processar(self):
        i = self.img_o.copy()
        self.find_contours(i)
        self.images_find(self.img_o.copy(), self.list_contours)
        self.drawHist(self.img_find_lig)
        
        self.bProcessar.config(state="disable")
        self.bOtsu.config(state="normal")
        self.limiarL.config(state="normal")
        self.limiarB.config(state="normal")
        
    def find_contours(self, img):
        i = img.copy()
        i = cv2.cvtColor(i, cv2.COLOR_RGB2LAB)
        a = i[:, :, 1]
        th = cv2.threshold(a,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        kernel = np.ones((5,5),np.uint8)
        i_close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        c_raw, h = cv2.findContours(i_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = self.filter_c(c_raw)
        
        i_marcada = img.copy()

        id_list = []
        count = 0
        for x in range(len(c)):
            M = cv2.moments(c[x])
            if (M['m00'] > 0):
                self.list_contours.append(c[x])

                M = cv2.moments(c[x])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                count+=1
                id_list.append(count)
                i_marcada = cv2.drawContours(i_marcada, c, x, (255, 0, 0), -1)
                i_marcada = cv2.putText(i_marcada, f"{count}", (cx-15, cy+15), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        
        self.id = id_list
        self.img_find_mrc = i_marcada.copy()
        self.showImg(self.img_find_mrc, self.lObjetos)
        self.lQuantidade.config(text= f"Partículas: {len(self.list_contours)}")

    def images_find(self, img, contours):
        imgL = img.copy()
        imgB = img.copy()
        img_black = np.zeros(shape=img.shape, dtype=img.dtype)
        
        i_mask = cv2.drawContours(img_black, contours, -1, (255, 255, 255), -1)[:, :, 1]
        self.img_find_agr = i_mask.copy()

        i_mask_inv = 255 - i_mask
        imgL[i_mask_inv == 255] = (255, 255, 255)
        imgB[i_mask_inv == 255] = (0, 0, 0)

        self.img_find_lig = imgL
        self.img_find_bri = imgB
    
    def drawHist(self, im):
        img = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        sorted_array = sorted(hist, reverse=True)
        maxHist = sorted_array[1][0]

        self.cHist.delete('all')
        for i in range(256):
            v = float(hist[i])
            v = (v/maxHist)*255
            v = 256 - v
            if(v <= 255):
                self.cHist.create_line(i, 255, i, v, fill='#ff0000', width=1)

    def sliderL(self, var):
        t = float(self.limiarL.get())
        self.cHist.delete('t1')
        self.cHist.create_line(t, 255, t, 0, fill='yellow', width=1, tag='t1')

        self.bLimiar.config(state="normal")
        self.bGravar.config(state="disable")
        self.bCalcular.config(state="disable")

    def sliderB(self, var):
        t = float(self.limiarB.get())
        self.cHist.delete('t2')
        self.cHist.create_line(t, 255, t, 0, fill='white', width=1, tag='t2')
        
        self.bGravar.config(state="disable")
        self.bCalcular.config(state="disable")

    def aplicando_limiar(self, imgL, imgB):
        ll = self.limiarL.get()
        lb = self.limiarB.get()
        il_cinza = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        ib_cinza = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
        retl, i_ll = cv2.threshold(il_cinza, ll, 255, cv2.THRESH_BINARY_INV)
        retb, i_lb = cv2.threshold(ib_cinza, lb, 255, cv2.THRESH_BINARY)

        i_o_l = self.img_o.copy()
        i_o_b = self.img_o.copy()
        i_o_l[i_ll == 255] = (255, 0, 0)
        i_o_b[i_lb == 255] = (255, 0, 0)

        self.img_count_lig = i_ll
        self.img_count_bri = i_lb

        self.i_ligante = i_o_l
        self.i_brilho = i_o_b

        self.bCalcular.config(state="normal")
        self.bGravar.config(state="disable")
        self.showImg(i_o_l, self.lLigante)
        self.showImg(i_o_b, self.lBrilho)

    def aplicando_otsu(self, imgL, imgB):
        ll = self.limiarL.get()
        lb = self.limiarB.get()
        il_cinza = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        ib_cinza = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
        retl, i_ll = cv2.threshold(il_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        retb, i_lb = cv2.threshold(ib_cinza, lb, 255, cv2.THRESH_BINARY)

        t = float(retl)
        self.limiarL.set(t)
        self.cHist.delete('t1')
        self.cHist.create_line(t, 255, t, 0, fill='yellow', width=1, tag='t1')

        i_o_l = self.img_o.copy()
        i_o_b = self.img_o.copy()
        i_o_l[i_ll == 255] = (255, 0, 0)
        i_o_b[i_lb == 255] = (255, 0, 0)

        self.img_count_lig = i_ll
        self.img_count_bri = i_lb

        self.i_ligante = i_o_l
        self.i_brilho = i_o_b

        self.bCalcular.config(state="normal")
        self.bGravar.config(state="disable")
        self.showImg(i_o_l, self.lLigante)
        self.showImg(i_o_b, self.lBrilho)

    def filter_c(self, contours_c):
        
        lenghts = []

        for i in contours_c:
            lenghts.append(len(i))

        size1 = 0.1 * (np.amax(lenghts))
        threshold = size1
        
        c_copy = np.copy(contours_c)
        for i, c in enumerate(contours_c):
            if lenghts[i] < threshold:
                c_copy[i] = None

        c_filtered = [c for c in c_copy if c is not None]
        return c_filtered

    def calcular(self):
        self.contar_pixel(self.list_contours, self.img_find_agr, 1)
        self.contar_pixel(self.list_contours, self.img_count_lig, 2)
        self.contar_pixel(self.list_contours, self.img_count_bri, 3)

        cobrimento = []
        statistic = []
        for i in range(len(self.nAgreg)):
            cobrimento.append(100*(self.nLigan[i] + self.nBrilh[i])/self.nAgreg[i])
            statistic.append('')

        self.nCobrimento = cobrimento
        a_cobrimento = np.asarray(cobrimento)
        
        media = np.average(a_cobrimento)
        desv_pad = np.std(a_cobrimento)
        self.valor_limiar_lig = float(self.limiarL.get())
        self.valor_limiar_bri = float(self.limiarB.get())
        statistic[0] = f"{media}"
        statistic[1] = f"{desv_pad}"
        statistic[2] = f"{self.valor_limiar_lig}"
        statistic[3] = f"{self.valor_limiar_bri}"
        self.estatistic = statistic

        self.lCalculos.config(text=f"Quantidade de pixels\nAgregado: {sum(self.nAgreg)} | Ligante: {sum(self.nLigan)} | Brilho: {sum(self.nBrilh)}")
    
        self.bGravar.config(state="normal")
        self.bCalcular.config(state="disable")
        
        self.dispersion(self.nCobrimento)

    def gravar(self):
        pasta_armazenar = str(self.ePasta.get()).strip()
        if(len(pasta_armazenar) > 0): #removed  and not self.tem_caracter_especial(pasta_armazenar)
            self.ePasta.config(bg="#90ee90")
            img_or = cv2.cvtColor(self.img_o.copy(), cv2.COLOR_RGB2BGR)
            img_ma = cv2.cvtColor(self.img_find_mrc.copy(), cv2.COLOR_RGB2BGR)
            img_li = cv2.cvtColor(self.i_ligante.copy(), cv2.COLOR_RGB2BGR)
            img_br = cv2.cvtColor(self.i_brilho.copy(), cv2.COLOR_RGB2BGR)

            legenda = []
            for x in range(len(self.list_contours)):
                legenda.append('')
            
            legenda[0] = 'Media'
            legenda[1] = 'Desv. Pad.'
            legenda[2] = 'Limiar Ligante'
            legenda[3] = 'Limiar Brilho do ligante'

            agora = datetime.datetime.now()
            data_hora_formatada = agora.strftime("%d-%m-%Y_%H-%M-%S")
            df = pd.DataFrame({'Id': self.id, 'Pixels Agregados': self.nAgreg, 'Pixels Ligante': self.nLigan, 'Pixels Brilho': self.nBrilh, 'Cobrimento': self.nCobrimento, 'Estatistica': self.estatistic, 'Legenda': legenda})
            
            pasta2 = f"{pasta_armazenar}/{data_hora_formatada}"

            if not (os.path.exists(pasta2)):
                os.mkdir(pasta2)
            
            df.to_csv(pasta2 + '/data_' + data_hora_formatada + '.csv', index=False)

            try:
                cv2.imwrite(pasta2 + '/imageOriginal_' + data_hora_formatada + '.jpg', img_or)
                cv2.imwrite(pasta2 + '/imageMarcada_' + data_hora_formatada + '.jpg', img_ma)
                cv2.imwrite(pasta2 + '/imageLigante_' + data_hora_formatada + '.jpg', img_li)
                cv2.imwrite(pasta2 + '/imageBrilho_' + data_hora_formatada + '.jpg', img_br)
            except:
                self.lCalculos.config(text="ERRO AO GRAVAR IMAGEM")
        else:
            self.ePasta.config(bg="#ee9090")

    def contar_pixel(self, contours, i_bin, var_contada):
        i_b = np.zeros(shape=i_bin.shape, dtype=i_bin.dtype)
        var_ = []
        for x in range(len(contours)):
            i_mask = cv2.drawContours(i_b.copy(), contours, x, (255), -1)
            i_mask_inv = 255 - i_mask
            i_contagem = i_bin.copy()
            i_contagem[i_mask_inv == 255] = (0)
            var_.append(cv2.countNonZero(i_contagem))
        
        vars_dict = {1: "nAgreg", 2: "nLigan"}
        setattr(self, vars_dict.get(var_contada, "nBrilh"), var_)

    def tem_caracter_especial(self, string):
        pattern = r'[^a-zA-Z0-9\s]'  # Padrão para verificar se há caracteres especiais
        print(bool(re.search(pattern, string)))
        return bool(re.search(pattern, string))


    def dispersion(self, variable):
        data = variable.copy()
        size_array = len(data)
        x_axis = np.arange(1, size_array+1)

        data_mean = np.mean(data)
        data_std = np.std(data, ddof=1)
        data_cv = 100 * data_std / data_mean

        font_normal = {'family': 'Times New Roman', 'size': 12}
        font_bold = {'family': 'Times New Roman', 'size': 12, 'weight': 'bold'}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        #plotando dispersão
        fig_dispersion, ax_d = plt.subplots(1, 2, figsize=(7, 2.5), dpi=100)
        ax_d[0].scatter(x_axis, data, alpha=0.4, color='red', edgecolor='red', label=f'x̅={round(data_mean, 1)} | std={round(data_std, 1)}')
        ax_d[0].set_title('Dispersão', fontdict=font_bold)
        ax_d[0].legend(prop=font_legend)
        ax_d[0].set_xlabel('Partícula', fontdict=font_normal)
        ax_d[0].set_ylabel('%Cobrimento', fontdict=font_normal)
        ax_d[0].grid(False)
        ax_d[0].set_ylim(0, 100)
        ax_d[0].set_facecolor('#fff')
        ax_d[0].tick_params(axis='both', labelfontfamily='Times New Roman')

        #plotando média da dispersão
        x_media = np.linspace(0, int(size_array), 1000)
        y_media = np.ones_like(x_media) * data_mean
        ax_d[0].plot(x_media, y_media, color='red')

        #plotando histograma
        ax_d[1].hist(data, bins=10, density=True, alpha=0.4, color='red', edgecolor='red')
        ax_d[1].set_title('Histograma', fontdict=font_bold)
        ax_d[1].set_xlabel('%Cobrimento', fontdict=font_normal)
        ax_d[1].set_ylabel('Densidade de probabilidade', fontdict=font_normal)
        ax_d[1].grid(False)
        ax_d[1].set_xlim(0, 100)
        ax_d[1].set_facecolor('#fff')
        ax_d[1].tick_params(axis='both', labelfontfamily='Times New Roman')

        fig_dispersion.set_facecolor('#0000') 

        #plotando distribuição de weibull
        weibull_shape = (data_std / data_mean)**(-1.086)
        g = gamma(1+(1/weibull_shape))
        weibull_scale = data_mean/g

        x_wei = np.linspace(0, 100, 1000)
        curva_weibull_min = weibull_min.pdf(x_wei, weibull_shape, scale=weibull_scale)
        ax_d[1].plot(x_wei, curva_weibull_min, color='red', label='Weibull_min')
        ax_d[1].legend(prop=font_legend)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig_dispersion, master=self.fGraphics)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.config(background='#f0f0f0')
        canvas_widget.pack()

        # plt.show()

if __name__ == '__main__':
    MainClass()