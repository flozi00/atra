from moviepy.editor import *


def burn_subtitles(start, end, text):
    # Generate a text clip
    txt_clip = TextClip(
        text, fontsize=26, color="white", bg_color="black", method="caption"
    )

    txt_clip = txt_clip.set_pos("top")
    txt_clip = txt_clip.set_start(start)
    txt_clip = txt_clip.set_end(end)

    return txt_clip


def merge_subtitles(subtitles, base_file, audio_name):
    subtitle_frames = []
    for i in range(len(subtitles)):
        start = subtitles[i]["start_timestamp"]
        end = subtitles[i]["stop_timestamp"]
        text = subtitles[i]["text"]
        subtitle_frames.append(burn_subtitles(start, end, text))


    file_clip = VideoFileClip(base_file)
    subtitle_frames = [file_clip] + subtitle_frames
    audio = file_clip.audio

    composed_video = CompositeVideoClip(subtitle_frames, use_bgclip=True)
    composed_video.audio = audio

    composed_video.write_videofile(f"{audio_name}.mp4", fps=10, codec="libx264")

    return f"{audio_name}.mp4"
