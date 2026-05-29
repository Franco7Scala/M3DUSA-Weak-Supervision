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



