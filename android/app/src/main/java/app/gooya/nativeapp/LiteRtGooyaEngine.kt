package app.gooya.nativeapp

import android.content.Context

/** Strict asset gate for the fixed-voice LiteRT bundle; inference is enabled only after export. */
class LiteRtGooyaEngine(private val context: Context) {
    private val requiredAssets = listOf(
        "gooya/t3_prefill_q4.tflite",
        "gooya/t3_decode_q4.tflite",
        "gooya/s3gen_flow_q4.tflite",
        "gooya/s3gen_vocoder_q4.tflite",
        "gooya/runtime-manifest.json",
    )

    fun readiness(): Readiness {
        val available = context.assets.list("gooya")?.map { "gooya/$it" }?.toSet().orEmpty()
        val missing = requiredAssets.filterNot(available::contains)
        return if (missing.isEmpty()) Readiness.Ready else Readiness.Missing(missing)
    }

    sealed interface Readiness {
        data object Ready : Readiness
        data class Missing(val assets: List<String>) : Readiness
    }
}
