Sociologi NAAAAAAAAA



f1_score(data[target_type].y["ground_truth"].cpu().numpy(),data[target_type].y["ground_truth_surrogate"].cpu().numpy(), average='macro')





 # ### DOVE STANNO LE COSE (avrei voluto farti una print) ###
    # # gli embeddings stanno in data.embeddings (Nota: presi da news_all.csv e claim_all.csv cioè {target_type}_all.csv (colonna "embedding_roberta"). Si suppone che il csv sia fratello della cartella heterodata)
    # # la ground truth surrogata sta in data[target_type].y["ground_truth_surrogate"]
    # # la ground truth vera sta in data[target_type].y["ground_truth"]
    #
    # # ADAPTING MODEL TO DOMAIN
    # # loading and splitting data
    # x = torch.as_tensor(data.embeddings, dtype=torch.float32).to(device)
    # y = data[target_type].y["ground_truth"].to(device)
    # mask = data[target_type].train_mask.to(device)
    # masked_x = x[mask]
    # masked_y = y[mask]
    # dataset = TensorDataset(masked_x, masked_y)
    # # loading model
    # head_model = RobertaClassificationHead(hidden_size=768, num_labels=2).to(device)
    # head_model.load_state_dict(torch.load("/home/jovyan/projects/InfluentialNodes/models/classification_head.pt", map_location=device))
    # head_model.train()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(head_model.parameters(), lr=1e-4, weight_decay=0.01)
    # # training model
    # head_model = train(head_model, optimizer, criterion, 15, DataLoader(dataset, batch_size=32, shuffle=True), device)
    # # labeling data with fine-tuned model
    # unmasked_x = x[~mask]
    # inference_dataset = TensorDataset(unmasked_x)
    # inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False)
    # head_model.eval()
    # all_predictions = []
    # with torch.no_grad():
    #     for batch in inference_loader:
    #         x_batch = batch[0].to(device)
    #         logits = head_model(x_batch)
    #         preds = torch.argmax(logits, dim=1)
    #         all_predictions.append(preds.cpu())
    #
    # final_predictions = torch.cat(all_predictions)
    # data[target_type].y["ground_truth_surrogate"][~mask] = final_predictions.to(device)
    #








def remove_random_edges(data, percentage_to_remove):
    keep_ratio = 1.0 - (percentage_to_remove / 100.0)
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        num_edges = edge_index.size(1)
        if num_edges == 0:
            continue

        num_to_keep = int(round(num_edges * keep_ratio))
        perm = torch.randperm(num_edges)
        keep_indices = perm[:num_to_keep]
        data[edge_type].edge_index = edge_index[:, keep_indices]
        if "edge_attr" in data[edge_type]:
            data[edge_type].edge_attr = data[edge_type].edge_attr[keep_indices]

        if "edge_label" in data[edge_type]:
            data[edge_type].edge_label = data[edge_type].edge_label[keep_indices]

    return data