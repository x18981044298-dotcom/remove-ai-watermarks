"""Shared constants for AI metadata detection, C2PA parsing, and format support.

All modules reference these constants rather than hard-coding values,
so adding a new AI tool or metadata key requires updating only this file.
"""

from typing import NamedTuple

# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}

# AI-generated image metadata keys (Stable Diffusion, ComfyUI, Midjourney, etc.)
AI_METADATA_KEYS = [
    "parameters",  # Stable Diffusion WebUI (AUTOMATIC1111, Vladmandic)
    "postprocessing",  # SD WebUI post-processing info
    "extras",  # SD WebUI extras
    "workflow",  # ComfyUI workflow JSON
    "prompt",  # Some AI tools
    "Dream",  # DreamStudio
    "SD:mode",  # Stability AI
    "StableDiffusionVersion",  # SD version info
    "generation_time",  # Generation time info
    "Model",  # Model name
    "Model hash",  # Model hash
    "Seed",  # Seed value
]

# Standard PNG metadata keys
PNG_METADATA_KEYS = [
    "Author",
    "Title",
    "Description",
    "Copyright",
    "Creation Time",
    "Software",
    "Disclaimer",
    "Warning",
    "Source",
    "Comment",
]

# AI-related keywords for detection
AI_KEYWORDS = [
    "prompt",
    "negative_prompt",
    "sampler",
    "cfg_scale",
    "lora",
    "diffusion",
    "comfy",
    "midjourney",
    "dall-e",
    "dalle",
    "imagen",
    "firefly",
    "c2pa",
    "chatgpt",
    "gpt-4",
    "sora",
    "openai",
    "truepic",
    "stable_diffusion",
    "invokeai",
]

# C2PA (Coalition for Content Provenance and Authenticity) constants
# Used by Google Imagen, Adobe Firefly, Microsoft Designer, OpenAI, etc.
C2PA_CHUNK_TYPE = b"caBX"  # JUMBF container chunk type for C2PA
C2PA_SIGNATURES = [
    b"c2pa",
    b"C2PA",
    b"jumb",
    b"jumd",
    b"JUMBF",
    b"jumbf",
    b"cbor",
    b"contentcreds",
    b"digid",
    b"assertions",
    b"manifest",
]


# Single source of truth for every C2PA-signing vendor. The three per-vendor
# facts that used to live in separate tables -- the issuer byte signature
# (C2PA_ISSUERS), the SynthID pairing (SYNTHID_C2PA_ISSUERS), and the human
# platform label (identify._ISSUER_PLATFORM) -- are all fields here, so adding a
# new C2PA vendor is a single append below; the views derive automatically.
class C2paAiVendor(NamedTuple):
    issuer: bytes  # distinctive byte signature scanned in the manifest (cert org / signer)
    org: str  # resolved issuer/cert-org display name (the old C2PA_ISSUERS value)
    # Human platform label for identify; None marks a signing authority / non-generator
    # (e.g. Truepic), which never names an AI platform on its own.
    platform: str | None
    # Substring matched against the joined issuer-org names for platform attribution
    # (usually a shorter form of org, e.g. "Google" for "Google LLC"); None when platform is.
    needle: str | None
    synthid: bool = False  # vendor pairs an invisible SynthID pixel watermark with its C2PA manifest


# C2PA known vendors, ORDERED for first-match-wins platform attribution: when a
# manifest names several issuers (Microsoft Designer signs as "OpenAI, Microsoft"),
# the earlier entry wins so the product, not the backend engine, is named.
# Used by Google Imagen, Adobe Firefly, Microsoft Designer, OpenAI, etc.
C2PA_AI_VENDORS: tuple[C2paAiVendor, ...] = (
    # Microsoft signs both Designer and Bing Image Creator; Bing now runs its own
    # MAI-Image model (not DALL-E), so the label stays model-neutral.
    C2paAiVendor(b"Microsoft", "Microsoft", "Microsoft (Bing Image Creator / Designer)", "Microsoft"),
    C2paAiVendor(b"Adobe", "Adobe", "Adobe Firefly", "Adobe"),
    C2paAiVendor(b"OpenAI", "OpenAI", "OpenAI (ChatGPT / gpt-image / DALL-E / Sora)", "OpenAI", synthid=True),
    C2paAiVendor(b"Google", "Google LLC", "Google (Gemini / Imagen)", "Google", synthid=True),
    # Stability AI signs C2PA as "Stability AI" (cert org "Stability AI Ltd").
    # Verified on a live Brand Studio (DreamStudio successor) output, 2026-05-24.
    C2paAiVendor(b"Stability AI", "Stability AI", "Stability AI (Stable Image / DreamStudio)", "Stability AI"),
    # Black Forest Labs (FLUX) API output: claim_generator_info "Black Forest
    # Labs API" + a c2pa.ai_generated_content assertion + trainedAlgorithmicMedia.
    # Verified on a real signed FLUX JPEG, 2026-05-29.
    C2paAiVendor(b"Black Forest Labs", "Black Forest Labs", "Black Forest Labs (FLUX)", "Black Forest Labs"),
    # ByteDance's Volcano Engine (Volcengine) signs its AI image output with a
    # cert from certificate_center@volcengine.com -- the platform behind Doubao /
    # Jimeng. Verified on two real signed JPEGs, 2026-05-29.
    C2paAiVendor(
        b"volcengine", "ByteDance (Volcano Engine)", "ByteDance (Doubao / Jimeng / Volcano Engine)", "ByteDance"
    ),
    # Some Volcano Engine certs name the signer with the Chinese legal entity
    # "北京火山引擎科技有限公司" (Beijing Volcano Engine Technology Co., Ltd.) rather
    # than the latin "volcengine" -- the latin needle misses it entirely, so real
    # ByteDance output was un-attributed in production traffic. The issuer is the
    # UTF-8 of the Chinese name (it appears UTF-8-encoded in the manifest-store
    # JSON and the raw caBX bytes alike); it normalizes to the same "ByteDance"
    # needle and platform as the volcengine row, so the two collapse together for
    # clash detection. Verified against the mined retained corpus, 2026-06-20.
    C2paAiVendor(
        "北京火山引擎科技有限公司".encode(),
        "ByteDance (Volcano Engine)",
        "ByteDance (Doubao / Jimeng / Volcano Engine)",
        "ByteDance",
    ),
    # ByteDance's international brand (BytePlus / Seedream / Seededit) signs its
    # cert as "Byteplus Pte. Ltd." -- the bare ``volcengine`` needle misses it, so
    # real BytePlus AI output was mis-attributed (an incidental "Adobe XMP" string
    # in the file's XMP made it read "Adobe Firefly"). Adding the issuer means the
    # clean manifest issuer matches "BytePlus (ByteDance)" directly. The platform
    # string mirrors the volcengine row: both share the "ByteDance" needle, so the
    # earlier row's label wins anyway -- they normalize together for clash
    # detection. Verified on real signed files in production traffic, 2026-06-19.
    C2paAiVendor(b"Byteplus", "BytePlus (ByteDance)", "ByteDance (Doubao / Jimeng / Volcano Engine)", "ByteDance"),
    # Canva Magic Media signs AI-generated images as "Canva" with a generic
    # c2pa-rs claim generator + trainedAlgorithmicMedia; without this entry the
    # source read AI but no platform was attributed. Verified on real signed files
    # in production traffic, 2026-06-19. Canva does not use SynthID.
    C2paAiVendor(b"Canva", "Canva", "Canva (Magic Media)", "Canva"),
    # ElevenLabs is a pure generative-AI company (AI voice / audio, and image /
    # video via its API); it signs output as "Eleven Labs Inc.", so the C2PA
    # manifest alone marks AI generation. Verified against the mined retained
    # corpus, 2026-06-20. ElevenLabs does not use SynthID.
    C2paAiVendor(b"Eleven Labs", "ElevenLabs", "ElevenLabs", "ElevenLabs"),
    # Truepic is a C2PA signing authority, not an AI generator: no platform label,
    # never asserts is_ai (the verdict comes from the digital-source-type).
    C2paAiVendor(b"Truepic", "Truepic", None, None),
)

# Deliberately NOT registered as AI-generation vendors (mined-corpus candidates
# evaluated 2026-06-20):
#   - TikTok Inc.: signs C2PA as a content-provenance / AI-labeling authority on
#     uploads, not as an image generator. The is_ai verdict keys off the
#     digitalSourceType (trainedAlgorithmicMedia), which is already honored; a
#     bare TikTok signer marks distribution provenance, not generation, so adding
#     it as a generator needle would mis-label human uploads as AI.
#   - PixelBin.io (issuer "Fynd"): an image transformation / optimization / CDN
#     service. Its C2PA stamps a transform/upload step, not a generation event.
#   Both are excluded to avoid false-positive AI attribution; re-evaluate only
#   against a real signed file whose manifest carries a trainedAlgorithmicMedia
#   digital-source type produced by the vendor itself.

# Derived view -- add a vendor to C2PA_AI_VENDORS above, not here.
# C2PA issuer signature -> resolved org name, for the manifest byte-scan.
C2PA_ISSUERS: dict[bytes, str] = {v.issuer: v.org for v in C2PA_AI_VENDORS}

# C2PA issuers whose signed outputs also carry an invisible SynthID pixel
# watermark -- a metadata proxy for "SynthID is in the pixels":
#   - Google (Imagen/Gemini): embeds SynthID, long-standing (DeepMind docs).
#   - OpenAI (ChatGPT/Codex/API): pairs SynthID with C2PA since ~2026-05-20.
#     Confirmed by OpenAI's Help Center ("C2PA and SynthID in OpenAI-generated
#     images", updated 2026-05-21): "Images generated with ChatGPT, Codex, and
#     our API include both C2PA metadata and SynthID watermarks." OpenAI also
#     notes a signal may be absent if "the image was created before these
#     signals were available" -- so OpenAI images from BEFORE the rollout carry
#     C2PA WITHOUT SynthID (e.g. data/samples/openai-images-2/amur-leopard.png,
#     C2PA timestamp 2026-04-22). For OpenAI the proxy is therefore "likely",
#     not certain; the verdict string is hedged accordingly. OpenAI's own oracle
#     is openai.com/verify (Google's is the Gemini app "Verify with SynthID").
# The issuer byte ("OpenAI"/"Google") is verified locally against data/samples;
# the SynthID pairing is documented behavior (Google: DeepMind; OpenAI: above).
# Adobe Firefly and Microsoft Designer sign C2PA but do NOT use SynthID, so a
# C2PA manifest alone is not a SynthID signal -- the issuer is. The pixel
# watermark is not locally detectable (proprietary decoder); the C2PA companion
# is the proxy, and only while the manifest is intact.
# Derived from the `synthid` flag on C2PA_AI_VENDORS -- set it there, not here.
SYNTHID_C2PA_ISSUERS: frozenset[bytes] = frozenset(v.issuer for v in C2PA_AI_VENDORS if v.synthid)

# C2PA known AI tools
C2PA_AI_TOOLS = {
    b"GPT-4o": "GPT-4o",
    b"ChatGPT": "ChatGPT",
    b"Sora": "Sora",
    b"DALL-E": "DALL-E",
    b"DALL": "DALL-E",
    b"Imagen": "Imagen",
    b"Firefly": "Firefly",
}

# C2PA ``c2pa.soft-binding`` algorithm identifiers -> the forensic-watermark
# vendor that stamped the pixels. The manifest's ``alg`` field names the
# watermark scheme even when the watermark itself cannot be decoded locally, so
# a byte-scan for these (keyed on a distinctive prefix to catch all variants)
# tells us a third-party forensic watermark is present and whose. Verified
# against the official C2PA registry (github.com/c2pa-org/softbinding-algorithm-list).
# Adobe TrustMark is additionally decodable locally (see ``trustmark_detector``);
# the rest (Digimarc, Imatag, Steg.AI, etc.) are proprietary oracle-only decoders.
C2PA_SOFT_BINDINGS = {
    b"com.adobe.trustmark": "Adobe TrustMark",
    b"com.adobe.icn": "Adobe (content fingerprint)",
    b"com.digimarc": "Digimarc",
    b"com.imatag.lamark": "Imatag (Lamark)",
    b"ai.steg": "Steg.AI",
    b"com.microsoft.invismark": "Microsoft InvisMark",
    b"com.microsoft.wavmark": "Microsoft WavMark",
    b"com.verimatrix": "Verimatrix",
    b"com.nagra.nexguard": "NAGRA NexGuard",
    b"com.aiwatermark": "AIWatermark (Meta PixelSeal)",
    b"ai.trufo": "Trufo",
    b"app.overlai": "Overlai",
    b"com.markany": "MarkAny",
    b"com.mentaport": "Mentaport",
    b"es.lumatrace": "LumaTrace",
    b"ai.verda": "VerdaAI",
    b"ai.contentlens": "ContentLens",
    b"io.iscc": "ISCC (content code)",
}

# Lowercased substrings that mark an AI generator when found in an EXIF
# ``Software`` / XMP ``CreatorTool`` value. Conservative on purpose: plain
# editors like "Adobe Photoshop" or "GIMP" must NOT match (no AI token), so only
# generator names land here. Add new generators here, not inline.
AI_GENERATOR_TOKENS: frozenset[str] = frozenset(
    {
        "firefly",
        "dall-e",
        "dalle",
        "midjourney",
        "stable diffusion",
        "stable-diffusion",
        "stablediffusion",
        "comfyui",
        "automatic1111",
        "invokeai",
        "imagen",
        "gpt-image",
        "nightcafe",
        "ideogram",
        "leonardo",
        "flux",
        "dreamstudio",
    }
)

# C2PA action types
C2PA_ACTIONS = {
    b"c2pa.created": "created",
    b"c2pa.converted": "converted",
    b"c2pa.edited": "edited",
    b"c2pa.filtered": "filtered",
    b"c2pa.cropped": "cropped",
    b"c2pa.resized": "resized",
    b"c2pa.opened": "opened",
    b"c2pa.placed": "placed",
}

# PNG signature
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
