import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from model import HistPredictor, CNNEncoder, EventPainter,Transformer,ModelPos,ModelP,ModelT
from event_reading import read_event
import torchvision.transforms as transforms
import os
import random
from tqdm import tqdm

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
    xx = np.floor((event_stream[:,0]*33)).astype(int)
    yy= np.floor((event_stream[:,1]*33)).astype(int)
    pp = np.floor((event_stream[:,2])).astype(int)
    
    ii = np.where(pp == 2)
    if len(ii[0]) > 0 :
      np.add.at(hist, (pp[:ii[0][0]], yy[:ii[0][0]], xx[:ii[0][0]]), 1)
    else:
      np.add.at(hist, (pp, yy, xx), 1)
    return hist

def pad_N_MIST(N_MNIST,max_n_events):
  new_N_mist = np.zeros((max_n_events,4))
  new_N_mist[:,2] = 2

  ii = N_MNIST.shape[0] 
  new_N_mist[:ii ,:] = N_MNIST
  return new_N_mist
  
def create_loader(N_MNIST_dir, MNIST_dir,seed,batchsize,min_n_events, max_n_events,split):
  grayscale_transform = transforms.Grayscale()
  class_path_list = os.listdir(N_MNIST_dir)
  N_MNIST_list = []
  MNIST_list = []
  inputmap_list = []
  label_list = []
  nEvent_list = []
  for class_path in class_path_list:
      N_MNIST_class_path = os.path.join(N_MNIST_dir, class_path)
      MNIST_class_path = os.path.join(MNIST_dir, class_path)
      file_path_list = os.listdir(N_MNIST_class_path)
      for file_path in file_path_list:        
          N_MNIST_file_path = os.path.join(N_MNIST_class_path, file_path)
          N_MNIST = read_event(N_MNIST_file_path)
          nEvent_list.append(N_MNIST.shape[0])
          if N_MNIST.shape[0] < min_n_events or N_MNIST.shape[0] > max_n_events:
              1
              #print(f"ERROR: {N_MNIST_file_path}")
              continue
          N_MNIST = pad_N_MIST(N_MNIST,max_n_events)
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
  np.save('nEventList.npy', nEvent_list)
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

def train():  
  #SMALL
  '''
  N_MNIST_path = "./N_MNIST_small_training"
  MNIST_path = "./MNIST_img_small_training"
  batchsize = 32
  min_n_events = 100
  max_n_events = 390
  seed = 42
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,min_n_events,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  '''

  N_MNIST_train_path = "./NMNIST_Train"
  MNIST_train_path = "./MNIST_Train"
  N_MNIST_test_path = "./NMNIST_Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 32
  min_n_events = 100
  max_n_events = 390
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,min_n_events,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,min_n_events,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  
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
  for epoch in range(15000):
    #pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = np.array([0.,0.])
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:    
      x_maps = MNIST.to(device)
      y_event = N_MNIST.to(device)
      vis_feat = cnn_encoder(x_maps)
      predict_event = event_painter(vis_feat)

      loss_xy = loss_spatial(predict_event[:,:,:2], y_event[:,:,:2])

      std_loss_x = torch.abs(torch.std(N_MNIST[:,:,0])-torch.std(predict_event[:,:,0]))
      std_loss_y = torch.abs(torch.std(N_MNIST[:,:,1])-torch.std(predict_event[:,:,1]))
      loss_xy2 = std_loss_x + std_loss_y

      #loss_p = loss_polarity(predict_event[:,:,2], y_event[:,:,2])
      loss_t = loss_time(predict_event[:,:,3], y_event[:,:,3])
      loss_event_tot =  loss_xy + loss_xy2  + loss_t
      loss_event_tot.backward()
      optimizer.step()
      optimizer.zero_grad()
      #pbar_training.update(train_data_loader.batch_size)
      avg_train_loss += np.round([loss_xy.item()/len(train_data_loader)+loss_xy2.item()/len(train_data_loader), loss_t.item()/len(train_data_loader)],4)

    with torch.no_grad():
      avg_test_loss = np.array([0.,0.])
      for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
        x_maps = MNIST.to(device)
        y_event = N_MNIST.to(device)
        vis_feat = cnn_encoder(x_maps)
        predict_event = event_painter(vis_feat)
        loss_xy = loss_spatial(predict_event[:,:,:2], y_event[:,:,:2])

        std_loss_x = torch.abs(torch.std(N_MNIST[:,:,0])-torch.std(predict_event[:,:,0]))
        std_loss_y = torch.abs(torch.std(N_MNIST[:,:,1])-torch.std(predict_event[:,:,1]))
        loss_xy2 = std_loss_x + std_loss_y
      
        #loss_p = loss_polarity(predict_event[:,:,2], y_event[:,:,2])
        loss_t = loss_time(predict_event[:,:,3], y_event[:,:,3])
        #loss_event_tot =  loss_xy + loss_xy2  + loss_t
        #pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += np.round([loss_xy.item()/len(test_data_loader)+loss_xy2.item()/len(test_data_loader), loss_t.item()/len(test_data_loader)],4)


    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if (sum(avg_test_loss)) < min_loss:
      torch.save(cnn_encoder.state_dict(), "cnn_encoder.pt")
      torch.save(event_painter.state_dict(), "event_painter.pt")
      min_loss = sum(avg_test_loss)
    #pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) #,"loss histogram:",loss_histogram.item())\

    np.save("loss_list.npy",[train_loss_list, test_loss_list])
    torch.save(cnn_encoder.state_dict(), "cnn_encoder_end.pt")
    torch.save(event_painter.state_dict(), "event_painter_end.pt")

def train_multimodel():  
  #SMALL
  N_MNIST_path = "./N_MNIST_small_training"
  MNIST_path = "./MNIST_img_small_training"
  batchsize = 32
  min_n_events = 100
  max_n_events = 390
  seed = 42
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,min_n_events,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  
  #view_loader(train_data_loader)

  # init models
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))

  cnn_encoder = CNNEncoder().to(device)
  transformer = Transformer(max_n_events = max_n_events).to(device)
  modelPos = ModelPos().to(device)
  modelP = ModelP().to(device)
  modelT = ModelT().to(device)
  

  # init optimizer and loss
  params = list(cnn_encoder.parameters()) + list(transformer.parameters()) + list(modelPos.parameters()) + list(modelP.parameters()) + list(modelT.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0001)
  loss_spatial = nn.MSELoss()
  loss_polarity = nn.MSELoss()
  loss_time = nn.MSELoss()

  train_loss_list = []
  test_loss_list = []

  min_loss = 99999
  for epoch in range(8000):
    #pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = np.array([0.,0.,0.])
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:    
      x_maps = MNIST.to(device)
      y_event = N_MNIST.to(device)
      vis_feat = cnn_encoder(x_maps)
      vis_feat = transformer(vis_feat)

      predicted_pos = modelPos(vis_feat)
      loss_xy = loss_spatial(predicted_pos, y_event[:,:,:2])
      std_loss_x = torch.abs(torch.std(N_MNIST[:,:,0])-torch.std(predicted_pos[:,:,0]))
      std_loss_y = torch.abs(torch.std(N_MNIST[:,:,1])-torch.std(predicted_pos[:,:,1]))
      loss_xy2 = (std_loss_x + std_loss_y)/5

      predicted_p = modelP(vis_feat)
      #label_onehot = F.one_hot(y_event[:,:,2].to(torch.int64), 3).float()
      #loss_p = loss_polarity(predicted_p, label_onehot)
      loss_p = loss_time(predicted_p[:,:,0], y_event[:,:,2]/2)

      predicted_t = modelT(vis_feat)      
      loss_t = loss_time(predicted_t[:,:,0], y_event[:,:,3])

      loss_event_train =  loss_xy + loss_xy2 + loss_p + loss_t
      loss_event_train.backward()
      optimizer.step()
      optimizer.zero_grad()
      #pbar_training.update(train_data_loader.batch_size)
      avg_train_loss += np.round([loss_xy.item()/len(train_data_loader)+loss_xy2.item()/len(train_data_loader), loss_p.item()/len(train_data_loader), loss_t.item()/len(train_data_loader)],4)

    with torch.no_grad():
      avg_test_loss = np.array([0.,0.,0.])
      for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
        x_maps = MNIST.to(device)
        y_event = N_MNIST.to(device)
        vis_feat = cnn_encoder(x_maps)
        vis_feat = transformer(vis_feat)

        predicted_pos = modelPos(vis_feat)
        loss_xy = loss_spatial(predicted_pos, y_event[:,:,:2])
        std_loss_x = torch.abs(torch.std(N_MNIST[:,:,0])-torch.std(predicted_pos[:,:,0]))
        std_loss_y = torch.abs(torch.std(N_MNIST[:,:,1])-torch.std(predicted_pos[:,:,1]))
        loss_xy2 = (std_loss_x + std_loss_y)/5

        predicted_p = modelP(vis_feat)
        label_onehot = F.one_hot(y_event[:,:,2].to(torch.int64), 3).float()
        loss_p = loss_polarity(predicted_p, label_onehot)/10

        predicted_t = modelT(vis_feat)      
        loss_t = loss_time(predicted_t[:,:,0], y_event[:,:,3])

        loss_event_train =  loss_xy + loss_xy2 + loss_p + loss_t

        #pbar_training.update(train_data_loader.batch_size)
        avg_test_loss += np.round([loss_xy.item()/len(test_data_loader)+loss_xy2.item()/len(test_data_loader), loss_p.item()/len(test_data_loader), loss_t.item()/len(test_data_loader)],4)

    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if (sum(avg_test_loss)) < min_loss:
      torch.save(cnn_encoder.state_dict(), "cnn_encoder.pt")
      torch.save(transformer.state_dict(), "event_painter.pt")
      torch.save(modelPos.state_dict(), "modelPos.pt")
      torch.save(modelP.state_dict(), "modelP.pt")
      torch.save(modelT.state_dict(), "modelT.pt")
      min_loss = sum(avg_test_loss)
    #pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) #,"loss histogram:",loss_histogram.item())\

    np.save("loss_list.npy",[train_loss_list, test_loss_list])
    torch.save(cnn_encoder.state_dict(), "cnn_encoder.pt")
    torch.save(transformer.state_dict(), "event_painter.pt")
    torch.save(modelPos.state_dict(), "modelPos.pt")
    torch.save(modelP.state_dict(), "modelP.pt")
    torch.save(modelT.state_dict(), "modelT.pt")


def test():   
  N_MNIST_path = "./N_MNIST_small_training"
  MNIST_path = "./MNIST_img_small_training"

  batchsize = 32
  min_n_events = 100
  max_n_events = 390
  seed = 42
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,min_n_events,max_n_events,split=True)

  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  # init models
  cnn_encoder = CNNEncoder()
  event_painter = EventPainter(max_n_events = max_n_events)
  cnn_encoder.load_state_dict(torch.load("cnn_encoder.pt"))
  event_painter.load_state_dict(torch.load("event_painter.pt"))
  
  with torch.no_grad():
    index = random.randint(0,len(test_data_loader.dataset)-1)
    index = 1
    print(f"index:{index}")
    [y_event, MNIST, input_maps, label] = test_data_loader.dataset[index]
    x_maps = MNIST.reshape(1,MNIST.shape[0],MNIST.shape[1],MNIST.shape[2])
    vis_feat = cnn_encoder(x_maps)
    predict_event = event_painter(vis_feat)
    predict_event[0,:,2] = torch.round(predict_event[0,:,2])
    np.save("y_event.npy",y_event.numpy())
    np.save("predict_event.npy",predict_event.numpy())

  plt.subplot(2,2,1)
  plt.plot(y_event[:,0]*33, label="Real")
  plt.plot(predict_event[0,:,0]*33, alpha=0.6,label="Geneated")
  plt.title("x")
  plt.legend()

  plt.subplot(2,2,2)
  plt.plot(y_event[:,1]*33, label="Real")
  plt.plot(predict_event[0,:,1]*33, alpha=0.6,label="Geneated")
  plt.title("y")
  plt.legend()

  plt.subplot(2,2,3)
  plt.plot(y_event[:,2], '.',label="Real")

  ii = np.where(predict_event[0,:,2] == 2)
  acc = sum(predict_event[0,:,2] == y_event[:,2])/len(y_event[:,2])
  acc = np.round(acc.item(),4)
  if len(ii[0]) > 0 :
    acc1 = sum(predict_event[0,:ii[0][0],2] == y_event[:ii[0][0],2])/len(y_event[:,2])
    acc1 = np.round(acc1.item(),4)
  else:
    acc1 = acc

  #print((predict_event[0,:,0]*33))

  plt.plot(predict_event[0,:,2], '.',alpha=0.6,label="Geneated")
  print(acc1)
  print(acc)
  plt.title(f"p: acc = {acc}, acc1 = {acc1}")
  plt.legend()

  plt.subplot(2,2,4)
  plt.plot(y_event[:,3], label="Real")
  plt.plot(predict_event[0,:,3], alpha=0.6,label="Geneated")
  plt.title("Timestamp")
  plt.legend()

  plt.tight_layout()
  plt.savefig('parameter.png')
  plt.figure().clear()

  plt.subplot(1,3,1)
  histogram_alt = event2histogram_alt(predict_event[0])
  image = np.zeros((34, 34, 3)) 
  histogram = np.transpose(histogram_alt, (1, 2, 0))
  image[:,:,0:2] = histogram
  plt.imshow(image, cmap='magma')
  plt.title("Generated_alt")

  plt.subplot(1,3,2)
  histogram = event2histogram_alt(y_event)
  image = np.zeros((34, 34, 3)) 
  histogram = np.transpose(histogram, (1, 2, 0))
  image[:,:,0:2] = histogram
  plt.imshow(image, cmap='magma')
  plt.title("Real")

  plt.subplot(1,3,3)
  plt.imshow(MNIST[0], cmap='magma')
  plt.title("MNIST")

  plt.tight_layout()
  plt.savefig('histogram.png')
  plt.figure().clear()

  train_loss_list = np.load("loss_list.npy")
  plt.plot(train_loss_list[0,:,0],label = 'xy')
  #plt.plot(train_loss_list[0,:,1],label = 'p')
  #plt.plot(train_loss_list[0,:,2],label = 't')

  #plt.xlim([0,600])
  plt.legend()
  plt.tight_layout()
  plt.savefig('loss.png')


def test_multimodel():   
  N_MNIST_path = "./N_MNIST_small_training"
  MNIST_path = "./MNIST_img_small_training"

  batchsize = 32
  min_n_events = 100
  max_n_events = 390
  seed = 42
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,min_n_events,max_n_events,split=True)

  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  # init models
  cnn_encoder = CNNEncoder()
  transformer = Transformer(max_n_events = max_n_events)
  modelPos = ModelPos()
  modelP = ModelP()
  modelT = ModelT()
  
  cnn_encoder.load_state_dict(torch.load("cnn_encoder.pt"))
  transformer.load_state_dict(torch.load("event_painter.pt"))
  modelPos.load_state_dict(torch.load("modelPos.pt"))
  modelP.load_state_dict(torch.load("modelP.pt"))
  modelT.load_state_dict(torch.load("modelT.pt"))

  with torch.no_grad():
    index = random.randint(0,len(test_data_loader.dataset)-1)
    index = 2
    print(f"index:{index}")
    [y_event, MNIST, input_maps, label] = test_data_loader.dataset[index]
    x_maps = MNIST.reshape(1,MNIST.shape[0],MNIST.shape[1],MNIST.shape[2])

    vis_feat = cnn_encoder(x_maps)
    vis_feat = transformer(vis_feat)

    predicted_pos = modelPos(vis_feat)
    predicted_p = modelP(vis_feat)
    predicted_t = modelT(vis_feat)      
    predicted_p[0,:,:] = torch.round(predicted_p[0,:,:])
    
    np.save("y_event.npy",y_event.numpy())
    np.save("predict_event.npy",predicted_pos.numpy())

  plt.subplot(2,2,1)
  plt.plot(y_event[:,0]*33, label="Real")
  plt.plot(predicted_pos[0,:,0]*33, alpha=0.6,label="Geneated")
  plt.title("x")
  plt.legend()

  plt.subplot(2,2,2)
  plt.plot(y_event[:,1]*33, label="Real")
  plt.plot(predicted_pos[0,:,1]*33, alpha=0.6,label="Geneated")
  plt.title("y")
  plt.legend()

  plt.subplot(2,2,3)
  plt.plot(y_event[:,2], '.',label="Real")

  ii = np.where(predicted_p[0,:,:] == 2)
  acc = sum(predicted_p[0,:,0] == y_event[:,2])/len(y_event[:,2])
  acc = np.round(acc.item(),4)
  if len(ii[0]) > 0 :
    acc1 = sum(predicted_p[0,:ii[0][0]] == y_event[:ii[0][0],2])/len(y_event[:,2])
    acc1 = np.round(acc1.item(),4)
  else:
    acc1 = acc

  #print((predict_event[0,:,0]*33))

  plt.plot(predicted_p[0,:,0], '.',alpha=0.6,label="Geneated")
  print(acc1)
  print(acc)
  plt.title(f"p: acc = {acc}, acc1 = {acc1}")
  plt.legend()

  plt.subplot(2,2,4)
  plt.plot(y_event[:,3], label="Real")
  plt.plot(predicted_t[0,:,0], alpha=0.6,label="Geneated")
  plt.title("Timestamp")
  plt.legend()

  plt.tight_layout()
  plt.savefig('parameter.png')
  plt.figure().clear()

 
  plt.subplot(1,3,1)
  tmp = torch.cat((predicted_pos, predicted_p, predicted_t), 2)
  histogram_alt = event2histogram_alt(tmp[0])
  image = np.zeros((34, 34, 3)) 
  histogram = np.transpose(histogram_alt, (1, 2, 0))
  image[:,:,0:2] = histogram
  plt.imshow(image, cmap='magma')
  plt.title("Generated_alt")

  plt.subplot(1,3,2)
  histogram = event2histogram_alt(y_event)
  image = np.zeros((34, 34, 3)) 
  histogram = np.transpose(histogram, (1, 2, 0))
  image[:,:,0:2] = histogram
  plt.imshow(image, cmap='magma')
  plt.title("Real")

  plt.subplot(1,3,3)
  plt.imshow(MNIST[0], cmap='magma')
  plt.title("MNIST")

  plt.tight_layout()
  plt.savefig('histogram.png')
  plt.figure().clear()

  train_loss_list = np.load("loss_list.npy")
  plt.plot(train_loss_list[0,:,0],label = 'xy')
  #plt.plot(train_loss_list[0,:,1],label = 'p')
  #plt.plot(train_loss_list[0,:,2],label = 't')

  #plt.xlim([0,600])
  plt.legend()
  plt.tight_layout()
  plt.savefig('loss.png')


if __name__ == "__main__":
  #train()
  #train_multimodel()
  test()
  #test_multimodel()
  



