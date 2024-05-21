import torch

class Evaluation():
    def intersection_over_union(one_hot_truth, one_hot_predict):
        #sum = 0
        #classes = len(one_hot_truth)
        total_iou = 0
        valid_classes = 0
        
        for i in range(len(one_hot_truth)):
            
            intersection = one_hot_truth[i]*one_hot_predict[i]
            intersection = torch.sum(intersection).item()
            union = torch.sum(one_hot_truth[i]).item() + torch.sum(one_hot_predict[i]).item() - intersection
            
            if union == 0 :
                continue
            iou = intersection / union
            total_iou += iou
            valid_classes += 1
            
        if valid_classes == 0:
            return 0.0, 0.0
        
        mean_iou = total_iou / valid_classes
        return mean_iou, total_iou/len(one_hot_truth)
        
        
    def pixel_acc(label, pred):
        sum = 0
        channel, width, height = pred.size()
        eq = torch.eq(label,pred)
        sum += torch.sum(torch.all(eq, dim=0)).item()/(width*height)
        
        return (sum)
    
    def mean_acc(one_hot_truth, one_hot_label):
        sum = 0
        channels, width, height = one_hot_truth.size()
        for i in range(len(one_hot_truth)):
            true = torch.sum(torch.eq(one_hot_truth[i], 1) & torch.eq(one_hot_label[i], 1)).item()
            
            if (torch.sum(one_hot_truth[i]).item() == 0):
                sum += 1
                
            else :
                sum += (true/(torch.sum(one_hot_truth[i])).item())
                
        return (sum/channels)
    