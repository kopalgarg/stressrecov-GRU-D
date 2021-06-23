"""
Created on Sat May 12 16:48:54 2018

@author: Zhiyong
"""

def Train_Model(model, train_dataloader, valid_dataloader, num_epochs = 300, patience = 10, min_delta = 0.00001, learning_rate = 0.0001):
    print('Model Structure: ', model)
    print('Start Training ... ')

    model.cuda()
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, alpha=0.99)
    use_gpu = torch.cuda.is_available()
    
    interval = 1
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        losses_epoch_train = []
        losses_epoch_valid = []
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            model.zero_grad()

            outputs = model(inputs)
            
            if output_last:
                loss_train = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels))
            else:
                full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)
                loss_train = loss_MSE(outputs, full_labels)
        
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                
            model.zero_grad()
            
            outputs_val = model(inputs_val)
            
            if output_last:
                loss_valid = loss_MSE(torch.squeeze(outputs_val), torch.squeeze(labels_val))
            else:
                full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
                loss_valid = loss_MSE(outputs_val, full_labels_val)

            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid 
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        
        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
    plt.plot(losses_train, label='Train loss')
    plt.plot(losses_valid, label='Validation loss')            
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]


def Test_Model(model, test_dataloader, max_speed):
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
    else:
        output_last = model.output_last
    
    inputs, labels = next(iter(test_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    MAEs = []
    MAPEs = []

    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)
        
        outputs = model(inputs)
    
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        
        if output_last:
            loss_mse = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels))
            loss_l1 = loss_L1(torch.squeeze(outputs), torch.squeeze(labels))
            MAE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels)))
            MAPE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels)) / torch.squeeze(labels))
        else:
            loss_mse = loss_MSE(outputs[:,-1,:], labels)
            loss_l1 = loss_L1(outputs[:,-1,:], labels)
            MAE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels)))
            MAPE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels)) / torch.squeeze(labels))
            
        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
        MAEs.append(MAE.data)
        MAPEs.append(MAPE.data)
        
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    
    losses_l1 =torch.FloatTensor(losses_l1)
    losses_mse =torch.FloatTensor(losses_mse)
    MAEs =torch.FloatTensor(MAEs)
    MAPEs =torch.FloatTensor(MAPEs)
    #losses_l1 = np.array(losses_l1)
    #losses_mse = np.array(losses_mse)
    #MAEs = np.array(MAEs)
    #MAPEs = np.array(MAPEs)
    
    #mean_l1 = np.mean(losses_l1) * max_speed
    mean_l1 = torch.mean(losses_l1) * max_speed
    std_l1 = torch.std(losses_l1) * max_speed
    MAE_ = torch.mean(MAEs) * max_speed
    MAPE_ = torch.mean(MAPEs) * 100
    
    print('Tested: L1_mean: {}, L1_std: {}, MAE: {} MAPE: {}'.format(mean_l1, std_l1, MAE_, MAPE_))
    return [losses_l1, losses_mse, mean_l1, std_l1]