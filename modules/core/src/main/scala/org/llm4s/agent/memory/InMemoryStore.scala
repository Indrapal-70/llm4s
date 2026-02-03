package org.llm4s.agent.memory

import org.llm4s.error.NotFoundError
import org.llm4s.error.ValidationError
import org.llm4s.types.Result

import java.time.Instant

/**
 * In-memory implementation of MemoryStore.
 *
 * This implementation stores all memories in memory, making it
 * suitable for testing, short-lived agents, and scenarios where
 * persistence isn't required.
 *
 * Features:
 * - Fast lookups using indexed data structures
 * - Basic keyword search (semantic search requires embeddings)
 * - Immutable design (returns new store on every write)
 * - No external dependencies
 *
 * Limitations:
 * - Data is lost when the application terminates
 * - Memory usage grows with stored memories
 * - Keyword search is less sophisticated than vector search
 * - Not thread-safe without external synchronization (returns new store, so caller must handle atomic swaps)
 *
 * @param memories All stored memories indexed by ID
 * @param config Configuration options
 */
final case class InMemoryStore private (
  private val memories: Map[MemoryId, Memory],
  config: MemoryStoreConfig,
  private val embeddingService: Option[EmbeddingService] = None
) extends MemoryStore {

  override def store(memory: Memory): Result[MemoryStore] = {
    val updated = memories + (memory.id -> memory)

    // Check if we need cleanup
    config.maxMemories match {
      case Some(max) if updated.size > max =>
        // Remove oldest memories to stay under limit
        val toRemove    = updated.size - max
        val sorted      = updated.values.toSeq.sortBy(_.timestamp)
        val idsToRemove = sorted.take(toRemove).map(_.id).toSet
        Right(copy(memories = updated.filterNot { case (id, _) => idsToRemove.contains(id) }))

      case _ =>
        Right(copy(memories = updated))
    }
  }

  override def get(id: MemoryId): Result[Option[Memory]] =
    Right(memories.get(id))

  override def recall(
    filter: MemoryFilter,
    limit: Int
  ): Result[Seq[Memory]] = {
    val filtered = memories.values.filter(filter.matches).toSeq
    val sorted   = filtered.sortBy(_.timestamp)(Ordering[Instant].reverse)
    Right(sorted.take(limit))
  }

  override def search(
    query: String,
    topK: Int,
    filter: MemoryFilter
  ): Result[Seq[ScoredMemory]] = {
    if (query.trim.isEmpty) {
      return Right(Seq.empty)
    }
    // First filter by criteria
    val filtered = memories.values.filter(filter.matches).toSeq

    // Check if we have embeddings available
    val hasEmbeddings = filtered.exists(_.isEmbedded)

    if (hasEmbeddings) {
      (filtered.collect { case m if m.embedding.isDefined => m }, embeddingService) match {
        case (embeddedMemories, Some(service)) if embeddedMemories.nonEmpty =>
          service.embed(query).flatMap { queryEmbedding =>
            val candidates = embeddedMemories.flatMap { memory =>
              memory.embedding.flatMap { vector =>
                if (vector.length != queryEmbedding.length) {
                  None
                } else {
                  val similarity = VectorOps.cosineSimilarity(queryEmbedding, vector)
                  val score      = math.max(0.0, math.min(1.0, (similarity + 1.0) / 2.0))
                  Some(ScoredMemory(memory, score))
                }
              }
            }

            if (candidates.isEmpty) keywordSearch(query, filtered, topK)
            else Right(candidates.sorted(ScoredMemory.byScoreDescending).take(topK))
          }

        case _ =>
          keywordSearch(query, filtered, topK)
      }
    } else {
      keywordSearch(query, filtered, topK)
    }
  }

  /**
   * Simple keyword-based search scoring.
   * Tokenizes both query and content for symmetric term matching,
   * avoiding substring matches that can inflate scores.
   */
  private def keywordSearch(
    query: String,
    memories: Seq[Memory],
    topK: Int
  ): Result[Seq[ScoredMemory]] = {
    def tokenize(s: String): Set[String] =
      s.toLowerCase.split("[^\\p{L}\\p{N}]+").filter(_.nonEmpty).toSet

    val queryTerms = tokenize(query)
    if (queryTerms.isEmpty) {
      return Right(Seq.empty)
    }

    val scored = memories.map { memory =>
      val contentTerms = tokenize(memory.content)
      val matchedTerms = queryTerms.intersect(contentTerms).size
      val score        = matchedTerms.toDouble / queryTerms.size
      ScoredMemory(memory, score)
    }

    val sorted = scored
      .filter(_.score > 0)
      .sorted(ScoredMemory.byScoreDescending)
      .take(topK)

    Right(sorted)
  }

  override def delete(id: MemoryId): Result[MemoryStore] =
    Right(copy(memories = memories - id))

  override def deleteMatching(filter: MemoryFilter): Result[MemoryStore] =
    Right(copy(memories = memories.filterNot { case (_, memory) => filter.matches(memory) }))

  override def update(id: MemoryId, updateFn: Memory => Memory): Result[MemoryStore] =
    memories.get(id) match {
      case Some(memory) =>
        val updated = updateFn(memory)
        if (updated.id != id) {
          Left(
            ValidationError(
              "id",
              s"update function changed Memory ID from $id to ${updated.id}; IDs must remain constant"
            )
          )
        } else {
          Right(copy(memories = memories + (id -> updated)))
        }

      case None =>
        Left(NotFoundError(s"Memory not found: $id", id.value))
    }

  override def count(filter: MemoryFilter): Result[Long] =
    Right(memories.values.count(filter.matches).toLong)

  override def clear(): Result[MemoryStore] =
    Right(copy(memories = Map.empty))

  override def recent(limit: Int, filter: MemoryFilter): Result[Seq[Memory]] = {
    val filtered = memories.values.filter(filter.matches).toSeq
    val sorted   = filtered.sortBy(_.timestamp)(Ordering[Instant].reverse)
    Right(sorted.take(limit))
  }

  /**
   * Get all memories (for debugging/testing).
   */
  def all: Seq[Memory] = memories.values.toSeq

  /**
   * Get memory count.
   */
  def size: Int = memories.size
}

object InMemoryStore {

  /**
   * Create an empty in-memory store with default configuration.
   */
  def empty: InMemoryStore = InMemoryStore(Map.empty, MemoryStoreConfig.default)

  /**
   * Create an empty in-memory store with custom configuration.
   */
  def apply(config: MemoryStoreConfig): InMemoryStore =
    InMemoryStore(Map.empty, config)

  /**
   * Create an empty in-memory store with an embedding service.
   *
   * When provided, the store can perform embedding-based semantic search.
   */
  def withEmbeddingService(
    service: EmbeddingService,
    config: MemoryStoreConfig = MemoryStoreConfig.default
  ): InMemoryStore =
    InMemoryStore(Map.empty, config, Some(service))

  /**
   * Create a store pre-populated with memories.
   */
  def withMemories(memories: Seq[Memory]): Result[InMemoryStore] = {
    val store = empty
    memories
      .foldLeft[Result[InMemoryStore]](Right(store)) { (acc, memory) =>
        acc.flatMap { s =>
          s.store(memory).flatMap {
            case next: InMemoryStore => Right(next)
            case other => Left(ValidationError("store", s"Expected InMemoryStore but got ${other.getClass.getName}"))
          }
        }
      }
  }

  /**
   * Create a store for testing with small limits.
   */
  def forTesting(maxMemories: Int = 1000): InMemoryStore =
    InMemoryStore(MemoryStoreConfig.testing.copy(maxMemories = Some(maxMemories)))
}
