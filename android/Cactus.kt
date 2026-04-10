package com.cactus

fun interface CactusTokenCallback {
    fun onToken(token: String, tokenId: Int)
}

fun interface CactusLogCallback {
    fun onLog(level: Int, component: String, message: String)
}

data class CactusGrammarJsonSchemaOptions(
    val anyWhitespace: Boolean = true,
    val indent: Int = 2,
    val itemSeparator: String = ",",
    val keyValueSeparator: String = ":",
    val strictMode: Boolean = true,
    val maxWhitespaceCount: Int = 1,
)

private object CactusJNI {
    init {
        System.loadLibrary("cactus")
        nativeSetFramework()
    }

    @JvmStatic external fun nativeSetFramework()
    @JvmStatic external fun nativeGetLastError(): String
    @JvmStatic external fun nativeSetCacheDir(cacheDir: String)
    @JvmStatic external fun nativeSetAppId(appId: String)
    @JvmStatic external fun nativeTelemetryFlush()
    @JvmStatic external fun nativeTelemetryShutdown()
    @JvmStatic external fun nativeInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long
    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativeReset(handle: Long)
    @JvmStatic external fun nativeStop(handle: Long)
    @JvmStatic external fun nativeComplete(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?, grammar: Long): String
    @JvmStatic external fun nativePrefill(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, pcmData: ByteArray?): String
    @JvmStatic external fun nativeGrammarInitGBNF(gbnf: String, startSymbol: String?): Long
    @JvmStatic external fun nativeGrammarInitJSON(): Long
    @JvmStatic external fun nativeGrammarInitEmpty(): Long
    @JvmStatic external fun nativeGrammarInitUniversal(): Long
    @JvmStatic external fun nativeGrammarInitJSONSchema(jsonSchema: String, anyWhitespace: Boolean, indent: Int, itemSeparator: String, keyValueSeparator: String, strictMode: Boolean, maxWhitespaceCount: Int): Long
    @JvmStatic external fun nativeGrammarInitRegex(regex: String): Long
    @JvmStatic external fun nativeGrammarInitStructuralTag(structuralTagJson: String): Long
    @JvmStatic external fun nativeGrammarUnion(grammars: LongArray): Long
    @JvmStatic external fun nativeGrammarConcatenate(grammars: LongArray): Long
    @JvmStatic external fun nativeGrammarIsEmpty(grammar: Long): Boolean
    @JvmStatic external fun nativeGrammarIsUniversal(grammar: Long): Boolean
    @JvmStatic external fun nativeGrammarDestroy(grammar: Long)
    @JvmStatic external fun nativeTranscribe(handle: Long, audioPath: String?, prompt: String?, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String
    @JvmStatic external fun nativeEmbed(handle: Long, text: String, normalize: Boolean): FloatArray
    @JvmStatic external fun nativeRagQuery(handle: Long, query: String, topK: Int): String
    @JvmStatic external fun nativeTokenize(handle: Long, text: String): IntArray
    @JvmStatic external fun nativeScoreWindow(handle: Long, tokens: IntArray, start: Int, end: Int, context: Int): String
    @JvmStatic external fun nativeImageEmbed(handle: Long, imagePath: String): FloatArray
    @JvmStatic external fun nativeAudioEmbed(handle: Long, audioPath: String): FloatArray
    @JvmStatic external fun nativeVad(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String
    @JvmStatic external fun nativeDiarize(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String
    @JvmStatic external fun nativeEmbedSpeaker(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?, maskWeights: FloatArray?): String
    @JvmStatic external fun nativeStreamTranscribeInit(handle: Long, optionsJson: String?): Long
    @JvmStatic external fun nativeStreamTranscribeProcess(handle: Long, pcmData: ByteArray): String
    @JvmStatic external fun nativeStreamTranscribeStop(handle: Long): String
    @JvmStatic external fun nativeIndexInit(indexDir: String, embeddingDim: Int): Long
    @JvmStatic external fun nativeIndexAdd(handle: Long, ids: IntArray, documents: Array<String>, metadatas: Array<String>?, embeddings: Array<FloatArray>, embeddingDim: Long): Int
    @JvmStatic external fun nativeIndexDelete(handle: Long, ids: IntArray): Int
    @JvmStatic external fun nativeIndexGet(handle: Long, ids: IntArray): String
    @JvmStatic external fun nativeIndexQuery(handle: Long, embedding: FloatArray, topK: Long, optionsJson: String?): String
    @JvmStatic external fun nativeIndexCompact(handle: Long): Int
    @JvmStatic external fun nativeIndexDestroy(handle: Long)
    @JvmStatic external fun nativeDetectLanguage(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String
    @JvmStatic external fun nativeLogSetLevel(level: Int)
    @JvmStatic external fun nativeLogSetCallback(callback: CactusLogCallback?)
}

fun cactusInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long {
    val h = CactusJNI.nativeInit(modelPath, corpusDir, cacheIndex)
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize model" })
    return h
}
fun cactusDestroy(model: Long) = CactusJNI.nativeDestroy(model)
fun cactusReset(model: Long) = CactusJNI.nativeReset(model)
fun cactusStop(model: Long) = CactusJNI.nativeStop(model)
fun cactusGetLastError(): String = CactusJNI.nativeGetLastError()
fun cactusSetTelemetryEnvironment(cacheDir: String) = CactusJNI.nativeSetCacheDir(cacheDir)
fun cactusSetAppId(appId: String) = CactusJNI.nativeSetAppId(appId)
fun cactusTelemetryFlush() = CactusJNI.nativeTelemetryFlush()
fun cactusTelemetryShutdown() = CactusJNI.nativeTelemetryShutdown()
fun cactusComplete(model: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray? = null, grammar: Long? = null): String =
    CactusJNI.nativeComplete(model, messagesJson, optionsJson, toolsJson, callback, pcmData, grammar ?: 0L)
fun cactusPrefill(model: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, pcmData: ByteArray? = null): String =
    CactusJNI.nativePrefill(model, messagesJson, optionsJson, toolsJson, pcmData)
fun cactusGrammarInitGBNF(gbnf: String, startSymbol: String? = null): Long {
    val h = CactusJNI.nativeGrammarInitGBNF(gbnf, startSymbol)
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize GBNF grammar" })
    return h
}
fun cactusGrammarInitJSON(): Long {
    val h = CactusJNI.nativeGrammarInitJSON()
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize JSON grammar" })
    return h
}
fun cactusGrammarInitEmpty(): Long {
    val h = CactusJNI.nativeGrammarInitEmpty()
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize empty grammar" })
    return h
}
fun cactusGrammarInitUniversal(): Long {
    val h = CactusJNI.nativeGrammarInitUniversal()
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize universal grammar" })
    return h
}
fun cactusGrammarInitJSONSchema(jsonSchema: String, options: CactusGrammarJsonSchemaOptions = CactusGrammarJsonSchemaOptions()): Long {
    val h = CactusJNI.nativeGrammarInitJSONSchema(
        jsonSchema,
        options.anyWhitespace,
        options.indent,
        options.itemSeparator,
        options.keyValueSeparator,
        options.strictMode,
        options.maxWhitespaceCount,
    )
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize JSON schema grammar" })
    return h
}
fun cactusGrammarInitRegex(regex: String): Long {
    val h = CactusJNI.nativeGrammarInitRegex(regex)
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize regex grammar" })
    return h
}
fun cactusGrammarInitStructuralTag(structuralTagJson: String): Long {
    val h = CactusJNI.nativeGrammarInitStructuralTag(structuralTagJson)
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize structural tag grammar" })
    return h
}
fun cactusGrammarUnion(grammars: LongArray): Long {
    val h = CactusJNI.nativeGrammarUnion(grammars)
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to union grammars" })
    return h
}
fun cactusGrammarConcatenate(grammars: LongArray): Long {
    val h = CactusJNI.nativeGrammarConcatenate(grammars)
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to concatenate grammars" })
    return h
}
fun cactusGrammarIsEmpty(grammar: Long): Boolean =
    CactusJNI.nativeGrammarIsEmpty(grammar)
fun cactusGrammarIsUniversal(grammar: Long): Boolean =
    CactusJNI.nativeGrammarIsUniversal(grammar)
fun cactusGrammarDestroy(grammar: Long) =
    CactusJNI.nativeGrammarDestroy(grammar)
fun cactusTranscribe(model: Long, audioPath: String?, prompt: String?, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String =
    CactusJNI.nativeTranscribe(model, audioPath, prompt, optionsJson, callback, pcmData)
fun cactusEmbed(model: Long, text: String, normalize: Boolean): FloatArray =
    CactusJNI.nativeEmbed(model, text, normalize)
fun cactusImageEmbed(model: Long, imagePath: String): FloatArray =
    CactusJNI.nativeImageEmbed(model, imagePath)
fun cactusAudioEmbed(model: Long, audioPath: String): FloatArray =
    CactusJNI.nativeAudioEmbed(model, audioPath)
fun cactusVad(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String =
    CactusJNI.nativeVad(model, audioPath, optionsJson, pcmData)
fun cactusDiarize(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String =
    CactusJNI.nativeDiarize(model, audioPath, optionsJson, pcmData)
fun cactusEmbedSpeaker(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?, maskWeights: FloatArray? = null): String =
    CactusJNI.nativeEmbedSpeaker(model, audioPath, optionsJson, pcmData, maskWeights)
fun cactusRagQuery(model: Long, query: String, topK: Int): String =
    CactusJNI.nativeRagQuery(model, query, topK)
fun cactusTokenize(model: Long, text: String): IntArray =
    CactusJNI.nativeTokenize(model, text)
fun cactusScoreWindow(model: Long, tokens: IntArray, start: Int, end: Int, context: Int): String =
    CactusJNI.nativeScoreWindow(model, tokens, start, end, context)
fun cactusStreamTranscribeStart(model: Long, optionsJson: String?): Long {
    val h = CactusJNI.nativeStreamTranscribeInit(model, optionsJson)
    if (h == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to create stream transcriber" })
    return h
}
fun cactusStreamTranscribeProcess(stream: Long, pcmData: ByteArray): String =
    CactusJNI.nativeStreamTranscribeProcess(stream, pcmData)
fun cactusStreamTranscribeStop(stream: Long): String =
    CactusJNI.nativeStreamTranscribeStop(stream)
fun cactusIndexInit(indexDir: String, embeddingDim: Int): Long {
    val h = CactusJNI.nativeIndexInit(indexDir, embeddingDim)
    if (h == 0L) throw RuntimeException("Failed to initialize index")
    return h
}
fun cactusIndexAdd(index: Long, ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>?): Int =
    CactusJNI.nativeIndexAdd(index, ids, documents, metadatas, embeddings, embeddings[0].size.toLong())
fun cactusIndexDelete(index: Long, ids: IntArray): Int =
    CactusJNI.nativeIndexDelete(index, ids)
fun cactusIndexGet(index: Long, ids: IntArray): String =
    CactusJNI.nativeIndexGet(index, ids)
fun cactusIndexQuery(index: Long, embedding: FloatArray, optionsJson: String?): String =
    CactusJNI.nativeIndexQuery(index, embedding, 1000L, optionsJson)
fun cactusIndexCompact(index: Long): Int =
    CactusJNI.nativeIndexCompact(index)
fun cactusIndexDestroy(index: Long) =
    CactusJNI.nativeIndexDestroy(index)
fun cactusDetectLanguage(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String =
    CactusJNI.nativeDetectLanguage(model, audioPath, optionsJson, pcmData)
fun cactusLogSetLevel(level: Int) =
    CactusJNI.nativeLogSetLevel(level)
fun cactusLogSetCallback(callback: CactusLogCallback?) =
    CactusJNI.nativeLogSetCallback(callback)
