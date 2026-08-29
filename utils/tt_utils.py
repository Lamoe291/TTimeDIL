import torch


class prototype_pool():
     

     self.task_prototypes = {}
     self.task_covariance ={}


    def compute_task_prototype_and_covariance(model, dataloader, current_task_id, device=None, shrinkage=1e-5, use_session_adapted_task_prototypes=True):
        """
        Compute mean (prototype) and covariance for one task.
        Stores both prototype and precision matrix (Σ⁻¹).
        """
        print('compute prototypes...')
        
        device = device or next(model.parameters()).device

        model.eval()
        all_features = []
        with torch.no_grad():
            for x, _ in dataloader:  # ignore labels, only task-level
                x = x.to(device)
                if use_session_adapted_task_prototypes:
                    helper_weights = torch.zeros(current_task_id+1).to(device)
                    helper_weights[0] = 1
                    #task_weights = self.get_task_weights(distances=distances, weighting_strat='softmax')
                    #print('task_weights', task_weights)

                    feats = model.forward_features(x, task_weights=helper_weights)["x_norm_clstoken"]
                    #print(ret)
                    #feats = model.forward_features()
                else:
                    feats = model(x)
                all_features.append(feats)

        all_features = torch.cat(all_features, dim=0)  # [N, D]
        mean_vec = all_features.mean(dim=0, keepdim=True)  # [1, D]

        # Centered features
        centered = all_features - mean_vec
        # Covariance (D x D)
        cov = (centered.T @ centered) / (centered.shape[0] - 1)

        # Regularization for stability
        cov = cov + shrinkage * torch.eye(cov.shape[0]).to(device)

        # Inverse covariance (precision matrix)
        cov += 1e-6 * torch.eye(cov.size(0), device=cov.device)
        precision = torch.linalg.inv(cov)

        # Save
        return {'task_prototype': mean_vec.squeeze().to(device),
                'task_covariance': precision.to(device)}
