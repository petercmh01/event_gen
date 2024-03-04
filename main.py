import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from model import HistPredictor, CNNEncoder, EventPainter
from event_reading import read_event
import torchvision.transforms as transforms
import os
import random

def modify_file_name(file_path):
    nr = int(file_path.split('.')[0])
    new_path = str(nr-1)+".jpg"
    return new_path

def event2histogram(event_stream):
    hist = np.zeros((2, 34, 34))
    for event in event_stream.numpy():
        x = (33*event[0]).astype(int)
        y = (33*event[1]).astype(int)
        if event[2] == 1:
            hist[0, y, x] += 1
        else:
            hist[1, y, x] += 1
    return hist

def event2histogram_alt(event_stream):
    event_stream = event_stream.numpy()
    hist = np.zeros((2, 34, 34))
    xx = (event_stream[:,0]*33).astype(int)
    yy= (event_stream[:,1]*33).astype(int)
    pp = (event_stream[:,2]).astype(int)
    pp[pp==1] = 0
    np.add.at(hist, (pp, xx, yy), 1)
    return hist

def create_loader(N_MNIST_dir, MNIST_dir,seed,batchsize,max_n_events,split):
  grayscale_transform = transforms.Grayscale()
  class_path_list = os.listdir(N_MNIST_dir)
  N_MNIST_list = []
  MNIST_list = []
  inputmap_list = []
  label_list = []
  for class_path in class_path_list:
      N_MNIST_class_path = os.path.join(N_MNIST_dir, class_path)
      MNIST_class_path = os.path.join(MNIST_dir, class_path)
      file_path_list = os.listdir(N_MNIST_class_path)
      for file_path in file_path_list:        
          N_MNIST_file_path = os.path.join(N_MNIST_class_path, file_path)
          N_MNIST = read_event(N_MNIST_file_path)
          if N_MNIST.shape[0] < max_n_events:
              1
              #print(f"ERROR: {N_MNIST_file_path}")
              continue
          N_MNIST = N_MNIST[:max_n_events]
          N_MNIST = torch.tensor(N_MNIST).to(torch.float32)
          N_MNIST_list.append(N_MNIST)

          MNIST_file_path = os.path.join(MNIST_class_path, modify_file_name(file_path))
          MNIST = cv2.imread(MNIST_file_path)
          MNIST = cv2.resize(MNIST, (34, 34), interpolation=cv2.INTER_LINEAR)/255 
          MNIST = grayscale_transform(torch.tensor(MNIST).permute(2,0,1)).to(torch.float32)
          MNIST_list.append(MNIST)

          event_histogram_data = torch.tensor(event2histogram_alt(N_MNIST))
          input_map = torch.cat((MNIST, event_histogram_data), dim=0).to(torch.float32)
          inputmap_list.append(input_map)

          label_list.append(int(class_path))
  merged_data = list(zip(N_MNIST_list, MNIST_list, inputmap_list, label_list))
  random.seed(seed)
  random.shuffle(merged_data)

  if split == False:
    data_loader = torch.utils.data.DataLoader(
        dataset=merged_data,
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    return data_loader
  
  else:
    ii = int(0.7*len(merged_data))
    train_loader = torch.utils.data.DataLoader(
        dataset=merged_data[:ii],
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset= merged_data[ii:],
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    return train_loader,test_loader
     
def view_loader(data_loader):
    while 1:
      index = random.randint(0,len(data_loader.dataset)-1)
      data = data_loader.dataset[index]
      plt.subplot(1,2,1)
      image = np.zeros((34, 34, 3)) 
      histogram = np.transpose(event2histogram(data[0]), (1, 2, 0))
      image[:,:,0:2] = histogram
      plt.imshow(image, cmap='magma')
      plt.title(data[-1])

      plt.subplot(1,2,2)
      plt.imshow(data[1].permute(1,2,0), cmap='magma')
      plt.title(data[-1])
      plt.show()

'''
# load some event stream
# 写read event的时候只load了saccade1的event，目前还没考虑换了方向的运动
# 完整的data loader也没写）
event4 = read_event('./N_MNIST_small_training/4/00003.bin')
event7 = read_event('./N_MNIST_small_training/7/00030.bin')
event8 = read_event('./N_MNIST_small_training/8/00018.bin')

# cat event stream data; 960 event for each for now (3, 960,4)
event_stream_data = np.stack((event4[:960], event7[:960], event8[:960]))

# load the corrsonding images and upsample them to the same dimensions of event stream (34*34);
# matching: index - 1
img4 = cv2.imread('./MNIST_img_small_training/4/2.jpg')
img7 = cv2.imread('./MNIST_img_small_training/7/29.jpg')
img8 = cv2.imread('./MNIST_img_small_training/8/17.jpg')

img4 = cv2.resize(img4, (34, 34), interpolation=cv2.INTER_LINEAR)/255 #normalize to 0-1 after resize
img7 = cv2.resize(img7, (34, 34), interpolation=cv2.INTER_LINEAR)/255
img8 = cv2.resize(img8, (34, 34), interpolation=cv2.INTER_LINEAR)/255

img_data = np.stack((img4, img7, img8))


# get some event histogram (960 events for now)
hist = np.zeros((2, 34, 34))
list_hist = []

for event_stream in event_stream_data:
  for event in event_stream:
    x = (33*event[0]).astype(int)
    y = (33*event[1]).astype(int)
    if event[2] == 1:
      hist[0, y, x] += 1
    else:
      hist[1, y, x] += 1
  list_hist.append(hist)
  hist = np.zeros((2, 34, 34))

list_hist = np.array(list_hist)


# load data -> tensor

event_stream_data = torch.tensor(event_stream_data).to(torch.float32) #[N,NEvent,4]
event_histogram_data = torch.tensor(list_hist).to(torch.float32)
img_data = torch.tensor(img_data)
img_data = img_data.permute(0,3,1,2)  #[N,H,W.C] - > #[N,C,H,W]
grayscale_transform = transforms.Grayscale()
img_data = grayscale_transform(torch.Tensor(img_data))
input_maps = torch.cat((img_data, event_histogram_data), dim=1).to(torch.float32)

#plt.imshow(list_hist[1][0], cmap='magma')
#plt.show()

'''

def train():   
  N_MNIST_path = "./N_MNIST_small_training"
  MNIST_path = "./MNIST_img_small_training"
  batchsize = 32
  max_n_events = 960
  seed = 42
  train_data_loader,test_data_loader = create_loader(N_MNIST_path,MNIST_path,seed,batchsize,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")


  '''
  N_MNIST_train_path = "./NMNIST_Train/Train"
  MNIST_train_path = "./MNIST_Train"
  N_MNIST_test_path = "./NMNIST_Test/Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 16
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,batchsize,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,batchsize,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  '''

  #view_loader(train_data_loader)

  # init models
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))

  cnn_encoder = CNNEncoder().to(device)
  event_painter = EventPainter(max_n_events = max_n_events).to(device)

  # init optimizer and loss
  params = list(cnn_encoder.parameters()) + list(event_painter.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0001)
  loss_spatial = nn.MSELoss()
  loss_time = nn.MSELoss()
  loss_polarity = nn.MSELoss()

  train_loss_list = []
  test_loss_list = []

  min_loss = 99999
  for epoch in range(200):
    for [N_MNIS, MNIST, input_maps, label] in train_data_loader:    
      x_maps = input_maps.to(device)
      y_event = N_MNIS.to(device)
      vis_feat = cnn_encoder(x_maps)
      predict_event = event_painter(vis_feat)
      loss_xy = loss_spatial(predict_event[:,:,:2], y_event[:,:,:2])
      loss_p = loss_polarity(predict_event[:,:,2], y_event[:,:,2])
      loss_t = loss_time(predict_event[:,:,3], y_event[:,:,3])
      loss_event_train = loss_xy + loss_p + loss_t
      loss_event_train.backward()
      optimizer.step()
      optimizer.zero_grad()
      #List.append(loss_event.item())

    with torch.no_grad():
      for [N_MNIS, MNIST, input_maps, label] in test_data_loader:    
        x_maps = input_maps.to(device)
        y_event = N_MNIS.to(device)
        vis_feat = cnn_encoder(x_maps)
        predict_event = event_painter(vis_feat)
        loss_xy = loss_spatial(predict_event[:,:,:2], y_event[:,:,:2])
        loss_p = loss_polarity(predict_event[:,:,2], y_event[:,:,2])
        loss_t = loss_time(predict_event[:,:,3], y_event[:,:,3])
        loss_event_test = loss_xy + loss_p + loss_t
    
    train_loss_list.append(loss_event_train.item())
    test_loss_list.append(loss_event_test.item())
    if loss_event_test.item() < min_loss:
      torch.save(cnn_encoder.state_dict(), "cnn_encoder.pt")
      torch.save(event_painter.state_dict(), "event_painter.pt")
      min_loss = loss_event_test.item()
    print("epoch:", epoch,"loss_train:",loss_event_train.item(),"loss_test:",loss_event_test.item(),"loss_min:",min_loss) #,"loss histogram:",loss_histogram.item())
  print("train_loss_list=")
  print(train_loss_list)
  print("test_loss_list=")
  print(test_loss_list)

def test():   
  N_MNIST_path = "./N_MNIST_small_training"
  MNIST_path = "./MNIST_img_small_training"
  batchsize = 32
  max_n_events = 960
  seed = 42
  train_data_loader,test_data_loader = create_loader(N_MNIST_path,MNIST_path,seed,batchsize,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  # init models
  cnn_encoder = CNNEncoder()
  event_painter = EventPainter(max_n_events = max_n_events)
  cnn_encoder.load_state_dict(torch.load("cnn_encoder.pt"))
  event_painter.load_state_dict(torch.load("event_painter.pt"))
  
  plt.subplot(2,2,1)
  with torch.no_grad():
    index = random.randint(0,len(train_data_loader.dataset)-1)
    [y_event, MNIST, input_maps, label] = train_data_loader.dataset[index]
    x_maps = input_maps.reshape(1,input_maps.shape[0],input_maps.shape[1],input_maps.shape[2])
    vis_feat = cnn_encoder(x_maps)
    predict_event = event_painter(vis_feat)

  plt.plot(y_event[:,3]*100000, label="Real Events Timestamps")
  plt.plot(predict_event[0,:,3]*100000, alpha=0.6,label="Geneated Events Timestamps")
  plt.title("Train:Event Timestamp (y, in μs) vs. Event Index (x)")
  plt.legend()

  plt.subplot(2,2,2)
  image = np.zeros((34, 34, 3)) 
  histogram = np.transpose(event2histogram(predict_event), (1, 2, 0))
  image[:,:,0:2] = histogram
  plt.imshow(image, cmap='magma')
  
  plt.subplot(2,2,3)
  with torch.no_grad():
    index = random.randint(0,len(test_data_loader.dataset)-1)
    [y_event, MNIST, input_maps, label] = train_data_loader.dataset[index]
    x_maps = input_maps.reshape(1,input_maps.shape[0],input_maps.shape[1],input_maps.shape[2])
    vis_feat = cnn_encoder(x_maps)
    predict_event = event_painter(vis_feat)

  plt.plot(y_event[:,3]*100000, label="Real Events Timestamps")
  plt.plot(predict_event[0,:,3]*100000, alpha=0.6,label="Geneated Events Timestamps")
  plt.title("Test:Event Timestamp (y, in μs) vs. Event Index (x)")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  test()
  



