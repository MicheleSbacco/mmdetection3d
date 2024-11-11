def group_training_dictionaries(original_dict_list):
    # Create list to store final data
    results_list = []

    # Create temporary variables to handle the numerical data
    total_loss_sum = 0.0
    training_count = 0
    
    # Process all the original dictionary
    for entry in original_dict_list:
        
        # Update the temporary variables if the entry is of type "training"
        if entry["type"] == "training":
            total_loss_sum += entry["total_loss"]
            training_count += 1
        
        # ...otherwise append the grouped "training" dictionary and also append the "validation one"
        else:
            # Group and append the "training" dictionary
            average_loss = total_loss_sum / training_count
            results_list.append({"type": "training", "total_loss": average_loss})
            # Reset the temporary variables
            total_loss_sum = 0.0
            training_count = 0
            # Append the "validation" dictionary
            results_list.append(entry)

    # Correction if there are some "unused" training keys
    if training_count > 0:
        # Group and append the "training" dictionary
        average_loss = total_loss_sum / training_count
        results_list.append({"type": "training", "total_loss": average_loss})

    # Return the "results" list
    return results_list
