import torch


def CLAS(logits, label, seq_len, criterion, device, is_topk=True):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        if is_topk:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            tmp = torch.mean(tmp).view(1)
        else:
            tmp = torch.mean(logits[i, :seq_len[i]]).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    instance_logits = torch.sigmoid(instance_logits)

    clsloss = criterion(instance_logits, label)
    return clsloss


def CENTROPY(logits, logits2, seq_len, device):
    instance_logits = torch.tensor(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        tmp1 = torch.sigmoid(logits[i, :seq_len[i]]).squeeze()
        tmp2 = torch.sigmoid(logits2[i, :seq_len[i]]).squeeze()
        loss = torch.mean(-tmp1.detach() * torch.log(tmp2))
        instance_logits = instance_logits + loss
    instance_logits = instance_logits/logits.shape[0]
    return instance_logits


def train(dataloader, model, optimizer, criterion, device, is_topk):
    with torch.set_grad_enabled(True):
        model.train()
        for i, (input, label) in enumerate(dataloader):

            import ipdb
            ipdb.set_trace()

            # print(input.shape)  # torch.Size([128, 200, 1152])

            seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
            # seq_len:  tensor([200, 103, 200, 200, 200,  39, 200, 123,  79, 135, 200, 102, 127, 119,
                # 94, 165,  78, 170,  76, 103, 162, 200, 125, 156, 103,  57, 200, 200,
                # 185, 129, 120, 200,  74, 200, 126,  60, 200, 200,  30, 200, 200, 200,
                # 137,  98, 200,  34,  81, 200, 200,  51, 200,  42, 136, 200,  22, 200,
                # 82,  96, 200, 200, 200, 200, 200, 165,  21, 108,  98, 103, 200, 200,
                # 36, 200,  97, 200,  81,  94,  76,  21,  27,  67,  81, 144,  51,  82,
                # 157, 200,  82, 200, 200,  45,  60, 200, 200,  76, 200, 200, 122, 200,
                # 200, 178,  81,  62, 200, 200,  32, 117,  85, 200, 199, 200, 200, 200,
                # 200,  77, 200, 200,  34,  55, 105,  26, 105, 200, 200, 200, 200,  78,
                # 124, 200])
            input = input[:, :torch.max(seq_len), :]

            input, label = input.float().to(device), label.float().to(device)
            logits, logits2 = model(input, seq_len)
            clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)
            clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)
            croloss = CENTROPY(logits, logits2, seq_len, device)

            total_loss = clsloss + clsloss2 + 5*croloss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
