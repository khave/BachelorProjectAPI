import cv2
import os

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img


def extract_images(video_path, amount_of_frames=0, target_size=None, crop=False):
    """
    Extracts images from a video file and saves them in a folder
    :param video_path: Path to the video file
    :param amount_of_frames: Amount of frames to extract
    :param target_size: Size of the extracted images

    :type video_path: str
    :type amount_of_frames: int
    :type target_size: tuple

    :return: None
    """

    if not video_path.endswith('.mp4'):
        return

    vidcap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    save_path = os.path.dirname(video_path)
    print(f"Extracting images from {video_name}...")

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = 5 # Set interval to every fifth frame if no amount_of_frames is given
    if amount_of_frames != 0:
        interval = int(total_frames / amount_of_frames) # Save each interval frame so we get amount_of_frames and the dataset is balanced (we do not want just the first amount_of_frames frames)
        print(f"Total frames: {total_frames}\nInterval: {interval}")

    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        if count % interval == 0:
            # Save frame 
            # Resize image
            if target_size is not None:
                if crop:
                    image = center_crop(image, target_size)
                else:
                    image = cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)
            cv2.imwrite(f"{save_path}/frame_{count}.jpg", image)
            #print(f"Saved frame {count}")
        count += 1
        if (count/interval) == amount_of_frames:
            break
    print(f"Done. Extracted {count/interval} frames")
    # Cleanup
    vidcap.release()
    os.remove(video_path)

    

def main():
    """
    Main function
    """
    #print("Creating training set...")
    folder = "//home/pi/share/SDU/6. Semester/Bachelor Projekt/Datasets/dataset 224x224 fully-built-15 [ALL sets] /train"
    # Extract images from all videos in the folder and subfolders
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".mp4"):
                extract_images(os.path.join(root, file), amount_of_frames=15, target_size=(224, 224))

    #print("Creating test set...")
    #folder = "//home/pi/share/SDU/6. Semester/Bachelor Projekt/Datasets/dataset 224x224 fully-built-test [ALL sets + bonsai]/test"
    # Extract images from all videos in the folder and subfolders
    #for root, dirs, files in os.walk(folder):
    #    for file in files:
    #        if file.endswith(".mp4"):
    #            extract_images(os.path.join(root, file), amount_of_frames=300, target_size=(224, 224))



if __name__ == "__main__":
    main()
