import colorsys
import csv
import shutil
import traceback
import logging
import pandas as pd
from pyecharts import Line, Bar
# from pyecharts import options as opts
from torch import nn
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_imgsz, is_ascii, check_version, check_font
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import scale_image
from ultralytics.yolo.utils.plotting import colors
from PIL import ImageDraw, ImageFont, Image
from PIL import __version__ as pil_version
from UIFunctions import *
from nets.yolo import YoloBody
from utils.utils import (get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from collections import defaultdict
from pathlib import Path
from util.capnums import Camera
from util.rtsp_win import Window
from charts import Chart_win
import numpy as np
import time
import json
import torch
import sys
import cv2
import os
import cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YoloPredictor(QObject):
    yolo2main_pre_img = Signal(np.ndarray)  # raw image signal
    yolo2main_res_img = Signal(np.ndarray)  # test result signal
    yolo2main_status_msg = Signal(str)  # Detecting/pausing/stopping/testing complete/error reporting signal
    yolo2main_fps = Signal(str)  # fps
    yolo2main_labels = Signal(dict)  # Detected target results (number of each category)
    yolo2main_progress = Signal(int)  # Completeness
    yolo2main_class_num = Signal(int)  # Number of categories detected
    yolo2main_target_num = Signal(int)  # Targets detected
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "classes_path": 'model_data/cls_classes.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        # ---------------------------------------------------------------------#
        "phi": 's',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        QObject.__init__(self)
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        show_config(**self._defaults)
        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        project = 'runs/detect'
        name = cfg.name
        self.save_dir = increment_path(Path(project) / name, exist_ok=False)

        # GUI args
        self.used_model_name = None  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = ''  # input source
        self.stop_dtc = False  # Termination detection
        self.continue_dtc = True  # pause
        self.save_res = False  # Save test results
        self.save_txt = False  # save label(txt) file
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar

        # Usable if setup is done
        self.model = None
        self.data = None  # data_dict
        self.imgsz = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.carCnt = 0
        self.motoCnt = 0
        self.personCnt = 0
        self.busCnt = 0
        self.truckCnt = 0
        self.outputDir=os.getcwd()+"/output/"
    # main for detect
    @torch.no_grad()
    def run(self):
        global save_path
        try:
            if cfg.verbose:
                logger.info('程序开始运行')

            # set model
            self.yolo2main_status_msg.emit('Loding Model...')
            if not self.model:
                self.generate(self.new_model_name)
                self.used_model_name = self.new_model_name

            # set source
            self.vid_path, self.vid_writer = self.setup_source(self.source if self.source is not None else cfg.source)
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
            count = 0  # run location frame
            start_time = time.time()  # used to calculate the frame rate
            batch = iter(self.dataset)
            while True:
                # Termination detection
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection terminated!')
                    break
                if self.used_model_name != self.new_model_name:
                    self.yolo2main_status_msg.emit('Change Model...')
                    self.generate(self.new_model_name)
                    self.used_model_name = self.new_model_name
                # pause switch
                if self.continue_dtc:
                    # time.sleep(0.001)
                    self.yolo2main_status_msg.emit('Detecting...')
                    batch = next(self.dataset)  # next data
                    self.batch = batch
                    path, im, im0s, vid_cap, s = batch
                    # Calculation completion and frame rate (to be optimized)
                    count += 1  # frame count +1
                    if vid_cap:
                        all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)  # total frames
                    else:
                        all_count = 1000
                    self.progress_value = int(count / all_count * 1000)  # progress bar(0~1000)
                    if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                        self.yolo2main_fps.emit(str(int(5 / (time.time() - start_time))))
                        start_time = time.time()
                    # preprocess
                    with self.dt[0]:
                        im, shape, image = self.preprocess(im0s)
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim
                    # inference
                    with self.dt[1]:
                        preds = self.model(im)
                        preds = decode_outputs(preds, self.input_shape)
                    # postprocess 
                    with self.dt[2]:
                        pass
                    n = len(im)
                    for i in range(n):
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}
                        p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                            else (path, im0s.copy())
                        p = Path(p)  # the source dir
                        label_str, top_label, image = self.postprocess(preds, shape, (p, im, im0), image)
                        target_nums += label_str
                        set1 = set(top_label)
                        class_nums += len(set1)
                        # save img or video result
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name), image)

                        # Send test results
                        im0 = np.asarray(image)
                        self.yolo2main_res_img.emit(im0)  # after detection
                        self.yolo2main_pre_img.emit(
                            im0s if isinstance(im0s, np.ndarray) else im0s[0])  # Before testing
                        # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres / 1000)  # delay , ms

                    self.yolo2main_progress.emit(self.progress_value)  # progress bar
                # Detection completed
                if count + 1 >= all_count:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection completed')
                    break
        except Exception as e:
            # traceback.print_exc()
            self.yolo2main_status_msg.emit('%s' % e)

    def generate(self, model):
        self.model = YoloBody(self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model or cfg.model
        self.model.load_state_dict(torch.load(model, map_location=device))
        self.model.eval()

    def setup_source(self, source):
        self.imgsz = check_imgsz(cfg.imgsz, stride=32, min_dim=2)  # check image size
        transforms = None
        self.dataset = load_inference_source(source=source,
                                             transforms=transforms,
                                             imgsz=self.imgsz,
                                             vid_stride=cfg.vid_stride,
                                             stride=32,
                                             auto=True)
        self.source_type = self.dataset.source_type
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs
        return self.vid_path, self.vid_writer

    def preprocess(self, im0s):

        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        if self.source_type.webcam:
            image_shape = np.array(np.shape(im0s[0])[0:2])
            img = Image.fromarray(im0s[0])
        else:
            image_shape = np.array(np.shape(im0s)[0:2])
            img = Image.fromarray(im0s)
        image = img.convert('RGB')
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
        return images, image_shape, image

    def postprocess(self, preds, image_shape, batch, image):
        p, im, im0 = batch
        results = non_max_suppression(preds, self.num_classes, self.input_shape,
                                      image_shape, self.letterbox_image, conf_thres=self.conf_thres,
                                      nms_thres=self.iou_thres)

        if results[0] is None:
            return image
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]
        if len(top_label) == 0:
            print('no detections')
        log_string = len(top_label)
        colors = Colors()
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        fake_result = {}
        fake_result["model_data"] = {"objects": []}
        for i, c in list(enumerate(top_label)):
            if c == 2:
                self.carCnt = self.carCnt + 1
            if c == 0:
                self.personCnt+=1
            if c == 3:
                self.motoCnt = self.motoCnt + 1
            if c == 5:
                self.busCnt=self.busCnt+1
            if c == 7:
                self.truckCnt = self.truckCnt + 1
            if self.save_res:
                predicted_class = self.class_names[int(c)]
                score = top_conf[i]
                box = top_boxes[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                xyxy_list = box.tolist()
                conf_list = '{:.2f}'.format(score)
                fake_result['model_data']['objects'].append({
                    "xmin": int(xyxy_list[0]),
                    "ymin": int(xyxy_list[1]),
                    "xmax": int(xyxy_list[2]),
                    "ymax": int(xyxy_list[3]),
                    "confidence": conf_list,
                    "name": predicted_class
                })
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors(c, True))
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors(c, True))
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
        print(json.dumps(fake_result, indent=2))
        return log_string, top_label, image

    def save_preds(self, vid_cap, idx, save_path, image):
        im0 = np.asarray(image)
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)
        self.save_csv_data()
        self.generate_chart()

    def save_csv_data(self):
        if os.path.exists(self.outputDir+"csv/"):
            shutil.rmtree(self.outputDir+"csv/")
        os.mkdir(self.outputDir+"csv/")
        with open(self.outputDir+"csv/count.csv",'a',newline='') as csvfile:
            writer=csv.writer(csvfile)
            name=['class','count']
            y=[]
            z=[]
            y.extend(['car','person','moto','bus','truck'])
            z.extend([int(self.carCnt),int(self.personCnt),int(self.motoCnt),int(self.busCnt),int(self.truckCnt)])
            writer.writerow(name)
            writer.writerows(zip(y,z))
            csvfile.close()

    def generate_chart(self):
        if os.path.exists(self.outputDir + "chart/"):
            pass
        else:
            os.mkdir(self.outputDir + "chart/")

        bar = Bar('检测目标数量分析图')
        if os.path.exists(self.outputDir + "csv/count.csv"):
            df=pd.read_csv(self.outputDir+"csv/count.csv")
            bar.add('value',list(df['class']),list(df['count']),is_label_show=True)
            bar.render(self.outputDir + 'chart/count.html')


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pth')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))  # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)  # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()  # Create a Yolo instance
        self.select_model = self.model_box.currentText()  # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.yolo_thread = QThread()  # Create yolo thread
        #
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))  # iou box
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed scroll bar

        # Prompt window initialization
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(self.chose_cam)  # chose_cam
        self.src_rtsp_button.clicked.connect(self.chose_rtsp)  # chose_rtsp
        self.label_button.clicked.connect(self.open_tools)
        self.chart_button.clicked.connect(self.f2)
        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)  # pause/start
        self.stop_button.clicked.connect(self.stop)  # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))  # left navigation button
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # top right settings button

        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            showimg = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            img = QImage(result.data, result.shape[1], result.shape[0], result.shape[2] * result.shape[1],
                         QImage.Format_RGB32)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status('Please select a video source before starting detection...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # start button
                self.save_txt_button.setEnabled(False)  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                self.show_status('Detecting...')
                self.yolo_predict.continue_dtc = True  # Control whether Yolo is paused
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)  # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # end process
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # end process
            self.pre_video.clear()  # clear image display
            self.res_video.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold,
                                              "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Load File：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.close_button, title='提示', text='请稍等，正在检测摄像头设备', time=2000, auto=True).exec()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet('''
                                                        QMenu {
                                                        font-size: 16px;
                                                        font-family: "Microsoft YaHei UI";
                                                        font-weight: light;
                                                        color:white;
                                                        padding-left: 5px;
                                                        padding-right: 5px;
                                                        padding-top: 4px;
                                                        padding-bottom: 4px;
                                                        border-style: solid;
                                                        border-width: 0px;
                                                        border-color: rgba(255, 255, 255, 255);
                                                        border-radius: 3px;
                                                        background-color: rgba(200, 200, 200,50);}
                                                        ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()
            # y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.yolo_predict.source = action.text()
                self.show_status('Loading camera：{}'.format(action.text()))

        except Exception as e:
            self.show_status('%s' % e)

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "http://admin:admin@192.168.195.15:8081"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    def open_tools(self):
        import os
        # self.close()
        os.system(r'python E:\labelImg-master\labelImg.py')

    # Save test result button--picture/video
    def f2(self):
        self.chart=Chart_win()
        self.chart.show()

    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.yolo_predict.save_res = True

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.yolo_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()  # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)  # start key recovery
        self.save_res_button.setEnabled(True)  # Ability to use the save button
        self.save_txt_button.setEnabled(True)  # Ability to use the save button
        self.pre_video.clear()  # clear image display
        self.res_video.clear()  # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.show_status('IOU Threshold: %s' % str(x / 100))
            self.yolo_predict.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x / 100)
            self.show_status('Conf Threshold: %s' % str(x / 100))
            self.yolo_predict.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms

    # change model
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pth')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState() == Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState() == Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
