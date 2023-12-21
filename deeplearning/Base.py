import  os
import  torch
import  utils
import  time
import  torch.nn    as      nn
import  pandas      as      pd
from    torch.optim import  lr_scheduler
from    tqdm        import  tqdm
from sklearn.metrics import accuracy_score

    
class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def normal_accuracy(pred,labels):    
    return ((pred.argmax(dim=1)==labels).sum()/len(labels))*100


def train(
    train_loader:torch.utils.data.DataLoader,
    val_loader  :torch.utils.data.DataLoader,
    model       :torch.nn.Module,
    model_name  :str,
    epochs      :int,
    load_saved_model    :bool,
    ckpt_save_freq      :int ,
    ckpt_save_path      :str ,
    ckpt_path           :str ,
    report_path         :str ,
    
    optimizer,
    criterion,
    
    
    
    
    tets_loader     :torch.utils.data.DataLoader,
    test_ealuate    :bool                           = False         ,
    device          :str                            = utils.device  ,
    sleep_time = 0.0,
    lr_schedulerr = None,
    lr_schedulerr_setting =0,
    validation_threshold = 80):

    # model       = model.to(device)
    # criterion   = nn.CrossEntropyLoss()

    if load_saved_model:
        model, optimizer = load_model(
                                      ckpt_path=ckpt_path, model=model, optimizer=optimizer
                                        )


    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_acc_till_current_batch"])
    
    max_Accu_validation_previous = 0
    
    for epoch in tqdm(range(1, epochs + 1)):
        acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        # model.train()
        mode = "train"
        
        
        loop_train = tqdm(
                            enumerate(train_loader, 1),
                            total=len(train_loader),
                            desc="train",
                            position=0,
                            leave=True
                        )
        accuracy_dum=[]

        # print(f'Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]["lr"]}')    
        for batch_idx, batch in loop_train:
            optimizer.zero_grad()
            batch = [_.to(utils.device) for _ in batch]
            x, labels = batch
            labels_pred = model(x).squeeze(1)#.reshape(4)

            # Ensure that the data type (dtype) matches the device
            labels, labels_pred = labels.to(utils.device), labels_pred.to(utils.device, dtype=torch.float)

            loss = criterion(labels_pred.float(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            acc1 = accuracy_score(labels.detach().cpu().numpy(),
                                  torch.round(labels_pred).detach().cpu().numpy())
            accuracy_dum.append(acc1)
            acc1 = sum(accuracy_dum)/len(accuracy_dum)
            
            loss_avg_train.update(loss.item(), x.size(0))
            
            # if batch_idx%sleep_time==0:
            #     time.sleep(1)
                
            if lr_schedulerr_setting == 0:
                lr_schedulerr.step()


            
            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "image_type":"original",
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": x.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_train_loss_till_current_batch":loss_avg_train.avg,
                 "avg_train_acc_till_current_batch":acc1,
                 "avg_val_loss_till_current_batch":None,
                 "avg_val_acc_till_current_batch":None},index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                accuracy_train="{:.4f}".format(acc1),
                refresh=True,
            )
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )
        if lr_schedulerr_setting == 0:  
            lr_schedulerr.step()

        model.eval()
        mode = "val"
        with torch.inference_mode():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            acc1 = 0
            total = 0
            accuracy_dum=[]
            for batch_idx, batch in loop_val:
                batch = [_.to(utils.device) for _ in batch]
                optimizer.zero_grad()
                batch = [_.to(utils.device) for _ in batch]
                x, labels = batch
                labels_pred = model(x).squeeze(1)#.reshape(4)

                # Ensure that the data type (dtype) matches the device
                labels, labels_pred = labels.to(utils.device), labels_pred.to(utils.device, dtype=torch.float)

                loss = criterion(labels_pred.float(), labels.float())
               
                acc1 = accuracy_score(labels.detach().cpu().numpy(),
                                    torch.round(labels_pred).detach().cpu().numpy())


                accuracy_dum.append(acc1)
                acc1 = sum(accuracy_dum)/len(accuracy_dum)

                loss_avg_val.update(loss.item(), x.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": x.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg,
                     "avg_val_acc_till_current_batch":acc1},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    accuracy_val="{:.4f}".format(acc1),
                    refresh=True,
                )
                
            max_Accu_validation = acc1
            if max_Accu_validation>validation_threshold and max_Accu_validation>max_Accu_validation_previous:
                torch.save(model.state_dict(), f"{report_path}/{model_name}_valid_acc {acc1}.pt")
                
            max_Accu_validation_previous = acc1 
                
        if test_ealuate==True:
            mode = "test"
            with torch.inference_mode():
                loop_val = tqdm(
                                enumerate(tets_loader, 1),
                                total=len(tets_loader),
                                desc="test",
                                position=0,
                                leave=True,
                                )
                accuracy_dum=[]
                for batch_idx, batch in loop_val:
                    batch = [_.to(utils.device) for _ in batch]
                    x, labels = batch
                    labels_pred = model(x).reshape(4)

                    # Ensure that the data type (dtype) matches the device
                    labels, labels_pred = labels.to(utils.device), labels_pred.to(utils.device, dtype=torch.float)

                    loss = criterion(labels_pred.float(), labels.float())
                    
                    
                    loss = criterion(labels_pred, labels)
                    
                    acc1 = accuracy_score(labels.detach().cpu().numpy(),
                                  torch.round(labels_pred).detach().cpu().numpy())
                    accuracy_dum.append(acc1)
                    acc1 = sum(accuracy_dum)/len(accuracy_dum)

            
                    loss_avg_val.update(loss.item(), x.size(0))
                    new_row = pd.DataFrame(
                        {"model_name": model_name,
                        "mode": mode,
                        "image_type":"original",
                        "epoch": epoch,
                        "learning_rate":optimizer.param_groups[0]["lr"],
                        "batch_size": x.size(0),
                        "batch_index": batch_idx,
                        "loss_batch": loss.detach().item(),
                        "avg_train_loss_till_current_batch":None,
                        "avg_train_acc_till_current_batch":None,
                        "avg_val_loss_till_current_batch":loss_avg_val.avg,
                        "avg_val_acc_till_current_batch":acc1},index=[0],)
                    
                    report.loc[len(report)] = new_row.values[0]
                    loop_val.set_description(f"test - iteration : {epoch}")
                    loop_val.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                        accuracy_val="{:.4f}".format(acc1),
                        refresh=True,
                    )    
            
        
    report.to_csv(f"{report_path}/{model_name}_report.csv")
    torch.save(model.state_dict(), report_path+'/'+model_name+'.pt')
    return model, optimizer, report