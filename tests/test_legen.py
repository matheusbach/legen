import legen


def test_transcription_vad_defaults_to_auto():
    args = legen.build_parser().parse_args(["--input_path", "video.mp4"])
    assert args.transcription_vad == "auto"


def test_parser_accepts_all_vad_values():
    for method in ("auto", "pyannote", "silero", "none", "disabled", "off"):
        args = legen.build_parser().parse_args(["--input_path", "video.mp4", "--transcription_vad", method])
        assert args.transcription_vad == method


def test_forced_silero_warning_is_emitted_once(capsys):
    legen._warn_if_silero_with_diarization(vad_method="silero", diarize=True, transcription_engine="whisperx")
    output = capsys.readouterr().out
    assert output.count("pyannote VAD is highly recommended") == 1


def test_auto_and_explicit_no_vad_do_not_warn(capsys):
    legen._warn_if_silero_with_diarization(vad_method="auto", diarize=True, transcription_engine="whisperx")
    legen._warn_if_silero_with_diarization(vad_method="none", diarize=True, transcription_engine="whisperx")
    legen._warn_if_silero_with_diarization(vad_method="silero", diarize=True, transcription_engine="whisper")
    assert capsys.readouterr().out == ""
