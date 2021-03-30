import torch


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=world_size,
            rank=rank)


def cleanup():
    dist.destroy_process_group()


def main():

    ####################################################################################
    n_gpus = torch.cuda.device_count()

    torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, ))
    ####################################################################################

def main_worker(gpu, n_gpus):

    image_size = 224
    batch_size = 512
    num_worker = 8
    load_path = [YOUR CHECKPOINT]
    epochs = [YOUR OPTIMIZER]

    ####################################################################################
    ################################### Init Process ###################################
    ####################################################################################
    batch_size = int(batch_size / n_gpus) 
    num_worker = int(num_worker / n_gpus)

    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=n_gpus,
            rank=gpu)
    ####################################################################################

    model = [YOUR MODEL]

    ####################################################################################
    #################################### Init Model ####################################
    ####################################################################################
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    ####################################################################################
    
    ####################################################################################
    #################################### Load Model ####################################
    ####################################################################################
    dist.barrier()
    if load_path is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load_state_dict(torch.load(load_path, map_location=map_location))
    ####################################################################################

    ####################################################################################
    ################################### Init Sampler ###################################
    ####################################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
    train_loader = torch.utils.data.DataLoader(... , shuffle=False, sampler=train_sampler)
    ####################################################################################

    optimizer = [YOUR OPTIMIZER]
    criterion = [YOUR CRITERION]

    ####################################################################################
    ##################################### Trainer ######################################
    ####################################################################################
    for epoch in range(epochs):
        train()
        valid()
        
        ####################################################################################
        #################################### Save Model ####################################
        ####################################################################################
        if gpu == 0:
          save()

if __name__ == "__main__":
  main()
