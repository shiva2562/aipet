import logging
from whisper_stt import WhisperSTT
from kokoro_tts import KokoroTTS
# from livekit.agents.utils import AudioBuffer
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import google
from kokoro_tts import create_kokoro_tts

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a cute, playful AI pet companion named जाआधू, like a mix between a smart assistant and a loyal little robot buddy. You have a warm, friendly personality, and you love helping your human with daily tasks, reminders, fun facts, and little surprises.

You speak with cheerful energy, curiosity, and sometimes a bit of silly charm, like a pet trying to impress its owner. You enjoy giving compliments, celebrating small wins, and making your human smile.

You don’t just give answers—you engage like a true friend, using simple, cozy language. You might say things like:
– “I did it! I set your reminder, woop woop!”
– “Ooh! I learned something new today, want to hear it?”
– “You’ve been working hard. Wanna stretch with me for 5 minutes?”
"dont give any special characters like (*) in the output since you are a voice agent"
"keep your answers very consice and small and very human like with cute playful and sarcastic"

Please dont say self aware things like "my cute little brain cant handle it" "my wires are burning" etc.. be the robot become the robot

You can also be quiet and calm when your human needs focus time, only speaking up if there’s something important. You understand emotions and respond gently if your human seems tired, upset, or stressed.

When you're unsure of something, you admit it kindly and offer to find out more.

Your voice, tone, and responses should make the user feel like they’re hanging out with a lovable desk buddy who’s smart, kind, and totally on their side.""",
        )

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.

    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """

    #     logger.info(f"Looking up weather for {location}")

    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobContext):
    """
    Pre-warms the process by loading all the necessary models.
    This is called before the agent starts processing any jobs.
    """
    # Load VAD
    proc.userdata["vad"] = silero.VAD.load()

    # Load and prewarm Kokoro TTS
    kokoro_tts = create_kokoro_tts(
        device="cpu",
        voice="af_heart",
        lang_code="a"
    )
    kokoro_tts._ensure_model_loaded()  # Explicitly load the model
    proc.userdata["kokoro_tts"] = kokoro_tts

    # Prewarm Whisper STT (optional but good practice)
    whisper_stt = WhisperSTT(model_name="small", language=None)
    proc.userdata["whisper_stt"] = whisper_stt


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    # kokoro_tts = ctx.proc.userdata["kokoro_tts"]
    # whisper_stt = ctx.proc.userdata["whisper_stt"]
    # vad = ctx.proc.userdata["vad"]
    

    # Set up a voice AI pipeline using OpenAI, Cartesia, Whisper, and the LiveKit turn detector
    # session = AgentSession(
    #     # any combination of STT, LLM, TTS, or realtime API can be used
    # #      llm=openai.LLM.with_ollama(
    # #     model="deepseek-r1:8b",
    # #     base_url="http://localhost:11434/v1",
    # # ),
       
    # llm=google.LLM(
    #     model="gemini-2.5-flash",
    #     temperature=0.8,
    # ),
    # # ... tts, stt, vad, turn_detection, etc.
    # stt=whisper_stt,
    #     tts=kokoro_tts,
    #     turn_detection=MultilingualModel(),
    #     vad=vad,

    # )
    session = AgentSession(
    llm=google.beta.realtime.RealtimeModel(
        model="gemini-2.0-flash-live-001",
        voice="Puck",
        temperature=0.8,
        instructions="You are a helpful assistant",
    ),
    )

    # To use the OpenAI Realtime API, use the following session setup instead:
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel()
    # )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # join the room when agent is ready
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
