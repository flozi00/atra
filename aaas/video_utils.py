from moviepy.editor import *

def burn_subtitles(start, end, text):    
    # Generate a text clip 
    txt_clip = TextClip(text, fontsize = 26, color = 'white', bg_color = "black", method = "caption") 
        
    txt_clip = txt_clip.set_pos('top')
    txt_clip = txt_clip.set_start(start)
    txt_clip = txt_clip.set_end(end)

    return txt_clip        


def merge_subtitles(subtitles, base_file, audio_name):
    subtitle_frames = []
    for i in range(len(subtitles)):
        start = subtitles[i]["timestamp"][0]
        end = subtitles[i]["timestamp"][1]
        text = subtitles[i]["text"]
        subtitle_frames.append(burn_subtitles(start, end, text))

    
    if base_file.split(".")[-1] not in ["mp4", "webm"]:
        audio_only = True
        file_clip = AudioFileClip(base_file)
    else:
        audio_only = False
        file_clip = VideoFileClip(base_file)
        subtitle_frames = [file_clip] + subtitle_frames
        audio = file_clip.audio

    if audio_only == True:
        composed_video = CompositeVideoClip(subtitle_frames, bg_color=[0x68, 0xAF, 0x14], size = (720,512))
        new_audioclip = CompositeAudioClip([file_clip])
        composed_video.audio = new_audioclip
    else:
        composed_video = CompositeVideoClip(subtitle_frames, use_bgclip=True)
        composed_video.audio = audio
    composed_video.write_videofile(f"{audio_name}.mp4", fps=10)

    return f"{audio_name}.mp4"