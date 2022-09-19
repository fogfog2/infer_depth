import cv2
import numpy as np
import torch
import os
from depthestimation.options_ucl import DepthOptions
from depthestimation.layers import transformation_from_parameters, disp_to_depth
from depthestimation import networks
from torchvision import transforms

class Inference():
    
    def __init__(self, opt):
        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        if opt.cuda_device is None:
            cuda_device = "cuda:0"
        else:
            cuda_device = opt.cuda_device
        
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_model =  opt.train_model
        
        if not opt.no_teacher:
            if "resnet" in encoder_model:            
                encoder_class = networks.ResnetEncoderMatching
            elif "swin" in encoder_model:
                encoder_class = networks.SwinEncoderMatching
            elif "cmt" in encoder_model:
                encoder_class = networks.CMTEncoderMatching
        else:
            if "resnet" in encoder_model:            
                encoder_class = networks.ResnetEncoder 
            elif "cmt" in encoder_model:
                encoder_class = networks.ResnetEncoderCMT
                
        encoder_dict = torch.load(encoder_path)
                
        try:
                HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                    'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width
        
        if "resnet" in encoder_model:            
            encoder_opts = dict(num_layers=opt.num_layers,
                            pretrained=False,
                            input_width=encoder_dict['width'],
                            input_height=encoder_dict['height'],
                            adaptive_bins=True,
                            min_depth_bin=0.1, max_depth_bin=20.0,
                            depth_binning=opt.depth_binning,
                            num_depth_bins=opt.num_depth_bins)
        elif "swin" in encoder_model:
            encoder_opts = dict(num_layers=opt.num_layers,
                            pretrained=False,
                            input_width=encoder_dict['width'],
                            input_height=encoder_dict['height'],
                            adaptive_bins=True,
                            min_depth_bin=0.1, max_depth_bin=20.0,
                            depth_binning=opt.depth_binning,
                            num_depth_bins=opt.num_depth_bins, use_swin_feature = opt.swin_use_feature)
        elif "cmt" in encoder_model:
            encoder_opts = dict(num_layers=opt.num_layers,
                            pretrained=False,
                            input_width=encoder_dict['width'],
                            input_height=encoder_dict['height'],
                            adaptive_bins=True,
                            min_depth_bin=0.1, max_depth_bin=20.0,
                            depth_binning=opt.depth_binning,
                            num_depth_bins=opt.num_depth_bins,
                            upconv = opt.cmt_use_upconv, start_layer = opt.cmt_layer, embed_dim = opt.cmt_dim,  use_cmt_feature = opt.cmt_use_feature
                            )
        
        pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
        pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))

        pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)

        pose_enc.load_state_dict(pose_enc_dict, strict=True)
        pose_dec.load_state_dict(pose_dec_dict, strict=True)

        min_depth_bin = encoder_dict.get('min_depth_bin')
        max_depth_bin = encoder_dict.get('max_depth_bin')

        pose_enc.eval()
        pose_dec.eval()
        
        if torch.cuda.is_available():
            pose_enc.cuda(cuda_device)
            pose_dec.cuda(cuda_device)
        
        self.encoder = encoder_class(**encoder_opts)     
        
        if opt.use_attention_decoder:            
            self.depth_decoder = networks.DepthDecoderAttention(self.encoder.num_ch_enc , no_spatial= opt.attention_only_channel)           
        else:
            self.depth_decoder = networks.DepthDecoder(self.encoder.num_ch_enc)
        
        model_dict = self.encoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        self.depth_decoder.load_state_dict(torch.load(decoder_path))

        self.encoder.eval()
        self.depth_decoder.eval()
        
        if torch.cuda.is_available():
            self.encoder.cuda(cuda_device)
            self.depth_decoder.cuda(cuda_device)
            
    
    def inference(self, opt,  input_color):       
            
        with torch.no_grad():
            output = self.encoder(input_color)
            output = self.depth_decoder(output)    
            
            pred_depth, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_depth = pred_depth.cpu()[:, 0].numpy()
        
        
        return pred_depth
    
if __name__ =="__main__":
    options = DepthOptions()
    infer = Inference(options.parse())            
    to_tensor = transforms.ToTensor()
 
    INPUT_MODE = ["VIDEO","IMAGE"]
    
    Mode = INPUT_MODE[1]
    
    if Mode==INPUT_MODE[0]:
        cap = cv2.VideoCapture(0)
    else:
        img_dir = "/home/sj/test_colon"
        filenames = sorted(os.listdir(img_dir))
    
    count = 0
    
    while True:                
        if Mode==INPUT_MODE[0]:
            ret, cv_image = cap.read()
        else:            
            
            if len(filenames)<=count:
                count = 0
            img_path = os.path.join(img_dir,filenames[count])            
            cv_image = cv2.imread(img_path)
            count=count+1            
                    
        tensor_image =  to_tensor(cv_image).unsqueeze(0)        
        
        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()
             
        depth = infer.inference(options.parse(),tensor_image)
        depth = np.clip(255-np.squeeze(depth)*60, 0, 255).astype(np.uint8)        
        color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        
        cv2.imshow("input",cv_image)        
        cv2.imshow("output",color)
        cv2.waitKey(33)
        
