from argparse import ArgumentParser
import os
import json
from get_args import Args
import torch
import numpy as np
import random
import utils
from models import myModel  # 修改版模型
# from model_old import myModel   # 对照实验：旧版模型
# from transformers import AdamW, get_linear_schedule_with_warmup

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import get_dataloader
import time
import warnings

# Only suppress specific known harmless warnings
warnings.filterwarnings("ignore", message=".*AdamW.*", category=FutureWarning)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import warnings
# 屏蔽transformers的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_optim(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer


def init_lr_scheduler(args, optim):
    t_total = args.epochs * args.episodes
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler


def deal_data(support_set, query_set, episode_labels):
    """Process episode data into model-ready format."""
    text, labels, flag = [], [], []

    # Ordered unique labels (preserving episode order)
    unique_list = list(dict.fromkeys(episode_labels))

    for idx, label in enumerate(unique_list):
        for x in support_set:
            if x["label"] == label:
                text.append(x["text"])
                labels.append(x["label"])
                flag.append(idx)

    for idx, label in enumerate(unique_list):
        for x in query_set:
            if x["label"] == label:
                text.append(x["text"])
                labels.append(x["label"])
                flag.append(idx)

    # One-hot encoding
    label_ids = []
    for label in labels:
        tmp = [1 if l == label else 0 for l in unique_list]
        label_ids.append(tmp)

    return text, labels, label_ids, flag, unique_list


def test(args, test_dataloader, model, modee, config_name):
    val_p, val_r, val_loss, val_f1, val_acc, val_auc = [], [], [], [], [], []
    logger = args.logger

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(test_dataloader):
            if i % 100 == 0:
                logger.info(f'--- test step {i} ---')

            support_set, query_set, episode_labels = batch
            text, labels, labels_ids, flag, unique_list = deal_data(support_set, query_set, episode_labels)
            loss, p, r, f, acc, auc = model(text, labels, labels_ids, flag, unique_list, modee="test")

            val_loss.append(loss.item())
            val_acc.append(acc)
            val_p.append(p)
            val_r.append(r)
            val_f1.append(f)
            val_auc.append(auc)

        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        avg_p = np.mean(val_p)
        avg_r = np.mean(val_r)
        avg_f1 = np.mean(val_f1)
        avg_auc = np.mean(val_auc)

        logger.info(f'Test Results - P: {avg_p:.4f}, R: {avg_r:.4f}, '
                    f'F1: {avg_f1:.4f}, Acc: {avg_acc:.4f}, AUC: {avg_auc:.4f}, Loss: {avg_loss:.4f}')
        print(f'\n{"="*50}')
        print(f'Final Test Results:')
        print(f'  Precision: {avg_p:.4f}')
        print(f'  Recall:    {avg_r:.4f}')
        print(f'  F1:        {avg_f1:.4f}')
        print(f'  Accuracy:  {avg_acc:.4f}')
        print(f'  AUC:       {avg_auc:.4f}')
        print(f'  Loss:      {avg_loss:.4f}')
        print(f'{"="*50}\n')

        path = args.save_path
        if not os.path.exists(path):
            os.makedirs(path)

        result_path = path + config_name
        with open(result_path, "a+") as fout:
            tmp = {"mode": modee, "p": avg_p, "r": avg_r, "f1": avg_f1,
                   "acc": avg_acc, "auc": avg_auc, "Loss": avg_loss}
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))


def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    p_best_state = None
    acc_best_state = None

    train_loss, epoch_train_loss = [], []
    train_acc, epoch_train_acc = [], []
    train_p, epoch_train_p = [], []
    train_r, epoch_train_r = [], []
    train_f1, epoch_train_f1 = [], []
    train_auc, epoch_train_auc = [], []

    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
    val_p, epoch_val_p = [], []
    val_r, epoch_val_r = [], []
    val_f1, epoch_val_f1 = [], []
    val_auc, epoch_val_auc = [], []

    best_p = 0
    best_acc = 0
    best_p_s = 0
    best_acc_s = 0
    logger = args.logger

    p_best_model_path = os.path.join(args.save_path, args.dataset + '_p_best_model.pth')
    acc_best_model_path = os.path.join(args.save_path, args.dataset + '_acc_best_model.pth')
    checkpoint_path = os.path.join(args.save_path, args.dataset + '_checkpoint.pth')

    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f'Resuming from checkpoint: {checkpoint_path}')
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_p = ckpt.get('best_p', 0)
        best_acc = ckpt.get('best_acc', 0)
        best_p_s = ckpt.get('best_p_s', 0)
        best_acc_s = ckpt.get('best_acc_s', 0)
        print(f'Resumed at epoch {start_epoch}, best_p={best_p:.4f}, best_acc={best_acc:.4f}')

    for epoch in range(start_epoch, args.epochs):
        print(f'\n=== Epoch: {epoch}/{args.epochs-1} ===')
        logger.info(f'=== Epoch: {epoch}/{args.epochs-1} ===')
        model.train()

        for i, batch in enumerate(tr_dataloader):
            optim.zero_grad()
            support_set, query_set, episode_labels = batch
            text, labels, labels_ids, flag, unique_list = deal_data(support_set, query_set, episode_labels)
            loss, p, r, f, acc, auc = model(text, labels, labels_ids, flag, unique_list, modee="train")

            loss.backward()
            optim.step()
            lr_scheduler.step()

            if i % 100 == 0:
                logger.info(f'  train step {i}/{args.episodes}')

            train_loss.append(loss.item())
            train_p.append(p)
            train_r.append(r)
            train_f1.append(f)
            train_acc.append(acc)
            train_auc.append(auc)

        avg_loss = np.mean(train_loss[-args.episodes:])
        avg_acc = np.mean(train_acc[-args.episodes:])
        avg_p = np.mean(train_p[-args.episodes:])
        avg_r = np.mean(train_r[-args.episodes:])
        avg_f1 = np.mean(train_f1[-args.episodes:])
        avg_auc = np.mean(train_auc[-args.episodes:])

        logger.info(f'Train - Loss: {avg_loss:.4f}, P: {avg_p:.4f}, R: {avg_r:.4f}, '
                    f'F1: {avg_f1:.4f}, Acc: {avg_acc:.4f}, AUC: {avg_auc:.4f}')

        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r)
        epoch_train_f1.append(avg_f1)
        epoch_train_auc.append(avg_auc)

        if val_dataloader is None:
            continue

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_dataloader):
                if i % 100 == 0:
                    logger.info(f'  val step {i}')

                support_set, query_set, episode_labels = batch
                text, labels, labels_ids, flag, unique_list = deal_data(support_set, query_set, episode_labels)
                loss, p, r, f, acc, auc = model(text, labels, labels_ids, flag, unique_list, modee="test")

                val_loss.append(loss.item())
                val_acc.append(acc)
                val_p.append(p)
                val_r.append(r)
                val_f1.append(f)
                val_auc.append(auc)

            avg_loss = np.mean(val_loss[-args.episodes:])
            avg_acc = np.mean(val_acc[-args.episodes:])
            avg_p = np.mean(val_p[-args.episodes:])
            avg_r = np.mean(val_r[-args.episodes:])
            avg_f1 = np.mean(val_f1[-args.episodes:])
            avg_auc = np.mean(val_auc[-args.episodes:])

            epoch_val_loss.append(avg_loss)
            epoch_val_acc.append(avg_acc)
            epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r)
            epoch_val_f1.append(avg_f1)
            epoch_val_auc.append(avg_auc)

        p_prefix = ' (Best)' if avg_p >= best_p else f' (Best:{best_p:.4f})'
        acc_prefix = ' (Best)' if avg_acc >= best_acc else f' (Best:{best_acc:.4f})'

        logger.info(f'Val - Loss: {avg_loss:.4f}, P: {avg_p:.4f}{p_prefix}, '
                    f'R: {avg_r:.4f}, F1: {avg_f1:.4f}, '
                    f'Acc: {avg_acc:.4f}{acc_prefix}, AUC: {avg_auc:.4f}')

        if avg_p >= best_p:
            torch.save(model.state_dict(), p_best_model_path)
            best_p = avg_p
            best_p_s = epoch
            p_best_state = model.state_dict()

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), acc_best_model_path)
            best_acc = avg_acc
            best_acc_s = epoch
            acc_best_state = model.state_dict()

        # Early stopping: only use metrics that are actually tracked
        last_p = epoch - best_p_s
        last_acc = epoch - best_acc_s
        last_change = max(last_p, last_acc)

        # Save checkpoint at end of each epoch for resumability
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_p': best_p,
            'best_acc': best_acc,
            'best_p_s': best_p_s,
            'best_acc_s': best_acc_s,
        }, checkpoint_path)

        if last_change >= 50:
            print(f'Early stopping at epoch {epoch} (no improvement for {last_change} epochs)')
            break

    # Save training curves
    for name in ['epoch_train_loss', 'epoch_train_p', 'epoch_train_r', 'epoch_train_f1', 'epoch_train_acc',
                 'epoch_train_auc', 'epoch_val_loss', 'epoch_val_p', 'epoch_val_r', 'epoch_val_f1', 'epoch_val_acc',
                 'epoch_val_auc']:
        utils.save_list_to_file(os.path.join(args.save_path, args.dataset + name + '.txt'), locals()[name])

    # Fallback: if p_best_state was never updated, use acc_best_state
    if p_best_state is not None:
        return p_best_state
    elif acc_best_state is not None:
        return acc_best_state
    else:
        return model.state_dict()


if __name__ == '__main__':
    parser = ArgumentParser(description="KFNpro: Variational Prototype Learning for Few-Shot Text Classification")
    parser.add_argument('--config', required=True, type=str, help="Path to JSON config file")
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--kshot', default=-1, type=int)
    parser.add_argument('--beta', default=-1.0, type=float)
    parser.add_argument('--temprature', default=-1.0, type=float)
    parser.add_argument('--tempra1', default=-1.0, type=float)
    parser.add_argument('--dataset_num', default="09", type=str)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--qshot', default=-1, type=int)
    parser.add_argument('--alpha', default=-1.0, type=float)
    parser.add_argument('--gama', default=-1.0, type=float)
    parser.add_argument('--output', default="no", type=str)
    parser.add_argument('--se_layer', default=-1, type=int)
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--margin', default=-1.0, type=float)
    parser.add_argument('--prompt_len', default=-1, type=int)
    parser.add_argument('--numFreeze', default=-1, type=int)
    parser.add_argument('--pool_len', default=-1, type=int)
    parser.add_argument('--epochs', default=-1, type=int)
    parser.add_argument('--warmup_steps', default=-1, type=int)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--learning_rate', default=-1.0, type=float)
    parser.add_argument('--dropout', default=-1.0, type=float)
    parser.add_argument('--gama1', default=-1.0, type=float)
    parser.add_argument('--gama2', default=-1.0, type=float)
    parser.add_argument('--dataset_name', default="no", type=str)
    parser.add_argument('--task', default="no", type=str)
    parser.add_argument('--text_len', default=-1, type=int)
    parser.add_argument('--label_len', default=-1, type=int)
    parser.add_argument('--optionn', default="bert_layer", type=str, help="mlp,nothing,bert_layer,mean")

    args_ = parser.parse_args()
    args = Args(args_.config)

    # Update config from CLI arguments
    args.update_from_args(args_)

    # Format paths with dataset number
    args.train_path = args.train_path.format(args.dataset)
    args.dev_path = args.dev_path.format(args.dataset)
    args.test_path = args.test_path.format(args.dataset)

    # Create output directory
    pathname = "../output/{}/test_{}_{}_{}".format(
        args_.dataset_name, args_.task, args_.output, time.strftime("%m-%d_%H-%M-%S"))
    os.makedirs(pathname, exist_ok=True)
    args.save_path = pathname
    args.checkpoint_path = f"run_{args_.dataset_name}_{args_.task}_{args_.output}_gpu_{args.gpu}"

    # Set seed
    set_seed(args.seed)

    # Save config
    config_name = '/' + args_.dataset_num + '_config.json'
    save_config_path = args.save_path + config_name
    args.write_self(save_config_path)
    args.show_self()

    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Load data
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader.get_loader(args)

    # Initialize model
    mymodel = myModel(args)
    mymodel.cuda()

    optim = init_optim(args, mymodel)
    lr_scheduler = init_lr_scheduler(args, optim)

    # Train
    results = train(
        args=args,
        tr_dataloader=train_dataloader,
        model=mymodel,
        optim=optim,
        lr_scheduler=lr_scheduler,
        val_dataloader=dev_dataloader
    )

    # Test with best model
    p_best_model_path = os.path.join(args.save_path, args.dataset + '_p_best_model.pth')
    acc_best_model_path = os.path.join(args.save_path, args.dataset + '_acc_best_model.pth')
    checkpoint_path = os.path.join(args.save_path, args.dataset + '_checkpoint.pth')

    mymodel.load_state_dict(results)
    print('Testing with best model...')
    test(args=args,
         test_dataloader=test_dataloader,
         model=mymodel,
         modee='best_p',
         config_name=config_name)

    # Clean up intermediate model files
    for path in [p_best_model_path, acc_best_model_path, checkpoint_path]:
        if os.path.exists(path):
            os.remove(path)
