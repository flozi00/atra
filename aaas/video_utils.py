from moviepy.editor import *

def burn_subtitles(start, end, text):    
    # Generate a text clip 
    txt_clip = TextClip(text, fontsize = 18, color = 'black', method = "caption") 
        
    txt_clip = txt_clip.set_pos('bottom')
    txt_clip = txt_clip.set_start(start)
    txt_clip = txt_clip.set_end(end)

    return txt_clip        


def merge_subtitles(subtitles, base_file):
    file_clip = AudioFileClip(base_file)
    subtitle_frames = []
    for i in range(len(subtitles)):
        start = subtitles[i]["timestamp"][0]
        end = subtitles[i]["timestamp"][1]
        text = subtitles[i]["text"]
        subtitle_frames.append(burn_subtitles(start, end, text))

    composed_video = CompositeVideoClip(subtitle_frames, bg_color=[0x68, 0xAF, 0x14], size = (720,512))
    new_audioclip = CompositeAudioClip([file_clip])
    composed_video.audio = new_audioclip
    composed_video.write_videofile("myvideo.mp4", fps=10)

    return "myvideo.mp4"