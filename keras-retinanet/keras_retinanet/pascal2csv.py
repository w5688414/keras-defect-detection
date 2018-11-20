import csv
import os
import glob
import sys
import xml.etree.ElementTree as ET

class PascalVOC2CSV(object):
    def __init__(self,root_dir, ann_path='./annotations.csv',classes_path='./classes.csv'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param ann_path: ann_path
        :param classes_path: classes_path
        '''
        anno_dir=os.path.join(root_dir,'Annotations/*.xml')
        xml_file = glob.glob(anno_dir)
        self.xml = xml_file
        self.ann_path = ann_path
        self.jpeg_path=os.path.join(root_dir,'JPEGImages')
        self.classes_path=classes_path
        self.label=[]
        self.annotations=[]

        self.data_transfer()
        self.write_file()


    def data_transfer(self):
        for num, xml_file in enumerate(self.xml):
            try:
                # print(xml_file)
                # 进度输出
                sys.stdout.write('\r>> Converting image %d/%d' % (
                    num + 1, len(self.xml)))
                sys.stdout.flush()
                tree=ET.ElementTree(file=xml_file)
                root=tree.getroot()
                self.file_name=root.find('filename').text
                ObjectSet=root.findall('object')
                for obj in ObjectSet:
                    ObjName=obj.find('name').text
                    bndBox=obj.find('bndbox')
                    x1=int(bndBox.find('xmin').text)
                    y1=int(bndBox.find('ymin').text)
                    x2=int(bndBox.find('xmax').text)
                    y2=int(bndBox.find('ymax').text)
                    
                    self.supercategory=ObjName
                    if self.supercategory not in self.label:
                        self.label.append(self.supercategory)
                    self.annotations.append([os.path.join(self.jpeg_path,self.file_name),x1,y1,x2,y2,self.supercategory])     
            except:
                continue

        sys.stdout.write('\n')
        sys.stdout.flush()

    def write_file(self,):
        with open(self.ann_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.annotations)

        class_name=sorted(self.label)
        class_=[]
        for num,name in enumerate(class_name):
            class_.append([name,num])
        with open(self.classes_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(class_)

root_dir='/home/eric/data/aihub/VOCdevkit/VOC2007/'
PascalVOC2CSV(root_dir)