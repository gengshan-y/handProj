Conf: 
{
  vidPath = "frame.mpg";         // folder name for output images, folder will be created
  // vidPath = "/data/gengshan/BUS/front1.m4v";
  // vidPath = "rtsp://admin:a123456@192.168.61.102";
  // vidPath = "/data/gengshan/MOT16/test/MOT16-01/img1/%06d.jpg";
  disp = 0;
  write = 1;
  dispRatio = 0.5;                // when displaying results
  
  pauseMs = 1;                    // for opencv wait key
  GPUID = 0;

  // detection
  det_conf_thresh = 0.8;    

  // tracking
  trk_age = 10;
  trk_score = 0.2;
};
