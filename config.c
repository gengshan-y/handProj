Conf: 
{
  // vidPath = "frame.mpg";         // folder name for output images, folder will be created
  // vidPath = "/data/gengshan/BUS/front1.m4v";
  vidPath = "rtsp://admin:a123456@192.168.61.102";
  disp = 1;
  write = 0;
  
  pauseMs = 0;                    // for opencv wait key
  GPUID = 0;

  // detection
  det_conf_thresh = 0.8;    

  // tracking
  trk_age = 10;
  trk_score = 0.2;
};
