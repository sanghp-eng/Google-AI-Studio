import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import {
  authMiddleware,
  AuthenticatedRequest,
  registerUser,
  loginUser,
  getDemoCredentials,
  ensureDemoUser,
  seedSampleKnowledgeBase,
} from './server/auth.js';
import {
  addDocument,
  getUserDocuments,
  getUserChunks,
  deleteDocument,
  searchVectorStore,
} from './server/vectorStore.js';
import { generateRAGAnswer } from './server/ragService.js';

dotenv.config();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '20mb' }));
  app.use(express.urlencoded({ extended: true, limit: '20mb' }));

  // Ensure default demo user exists
  await ensureDemoUser();

  // ----------------------------------------------------
  // Public Health & Auth Routes
  // ----------------------------------------------------
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      hasGeminiApiKey: !!process.env.GEMINI_API_KEY,
      timestamp: new Date().toISOString(),
    });
  });

  // Auth: Register
  app.post('/api/auth/register', async (req, res) => {
    try {
      const { email, password, name } = req.body;
      if (!email || !password) {
        return res.status(400).json({ error: 'Vui lòng nhập đầy đủ email và mật khẩu.' });
      }
      if (password.length < 6) {
        return res.status(400).json({ error: 'Mật khẩu phải có ít nhất 6 ký tự.' });
      }
      const result = await registerUser(email, password, name);
      res.json(result);
    } catch (err: any) {
      res.status(400).json({ error: err.message || 'Đăng ký thất bại' });
    }
  });

  // Auth: Login
  app.post('/api/auth/login', async (req, res) => {
    try {
      const { email, password } = req.body;
      if (!email || !password) {
        return res.status(400).json({ error: 'Vui lòng nhập email và mật khẩu.' });
      }
      const result = await loginUser(email, password);
      res.json(result);
    } catch (err: any) {
      res.status(401).json({ error: err.message || 'Đăng nhập thất bại' });
    }
  });

  // Auth: Demo Login
  app.post('/api/auth/demo', async (req, res) => {
    try {
      const result = getDemoCredentials();
      if (!result) {
        await ensureDemoUser();
        const retryResult = getDemoCredentials();
        return res.json(retryResult);
      }
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: 'Không thể khởi tạo tài khoản demo' });
    }
  });

  // Auth: Me
  app.get('/api/auth/me', authMiddleware, (req: AuthenticatedRequest, res) => {
    res.json({ user: req.user });
  });

  // ----------------------------------------------------
  // Protected Knowledge Base Routes
  // ----------------------------------------------------
  // Get all documents for current user
  app.get('/api/kb/documents', authMiddleware, (req: AuthenticatedRequest, res) => {
    try {
      const userId = req.user!.id;
      const docs = getUserDocuments(userId);
      res.json({ documents: docs });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // Add a new document (chunk and embed into vector store)
  app.post('/api/kb/documents', authMiddleware, async (req: AuthenticatedRequest, res) => {
    try {
      const userId = req.user!.id;
      const {
        title,
        content,
        category = 'Tài liệu chung',
        tags = [],
        chunkingStrategy = 'paragraph',
        chunkSize = 350,
        chunkOverlap = 50,
      } = req.body;

      if (!title || !content) {
        return res.status(400).json({ error: 'Tiêu đề và nội dung tài liệu không được để trống.' });
      }

      const { document, chunks } = await addDocument(
        userId,
        title,
        content,
        category,
        tags,
        chunkingStrategy,
        chunkSize,
        chunkOverlap
      );

      res.json({
        success: true,
        document,
        chunksCount: chunks.length,
        message: `Đã nạp và tạo vector embedding thành công cho ${chunks.length} phân đoạn (chunks).`,
      });
    } catch (err: any) {
      console.error('Error adding document:', err);
      res.status(500).json({ error: err.message || 'Lỗi khi xử lý nạp tài liệu và vector' });
    }
  });

  // Delete a document
  app.delete('/api/kb/documents/:id', authMiddleware, (req: AuthenticatedRequest, res) => {
    try {
      const userId = req.user!.id;
      const docId = req.params.id;
      const deleted = deleteDocument(userId, docId);
      if (deleted) {
        res.json({ success: true, message: 'Đã xóa tài liệu và các vector chunks liên quan.' });
      } else {
        res.status(404).json({ error: 'Không tìm thấy tài liệu cần xóa.' });
      }
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // Seed standard sample knowledge bases
  app.post('/api/kb/seed-presets', authMiddleware, async (req: AuthenticatedRequest, res) => {
    try {
      const userId = req.user!.id;
      await seedSampleKnowledgeBase(userId);
      const docs = getUserDocuments(userId);
      res.json({ success: true, documents: docs });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // Get user chunks
  app.get('/api/kb/chunks', authMiddleware, (req: AuthenticatedRequest, res) => {
    try {
      const userId = req.user!.id;
      const documentId = req.query.documentId as string | undefined;
      const chunks = getUserChunks(userId, documentId);
      // Return chunks with embedding dimensions without sending full 1536 float arrays to save bandwidth
      const lightChunks = chunks.map(c => ({
        id: c.id,
        documentId: c.documentId,
        documentTitle: c.documentTitle,
        category: c.category,
        chunkIndex: c.chunkIndex,
        content: c.content,
        tokenCount: c.tokenCount,
        characterCount: c.characterCount,
        embeddingDim: c.embedding.length,
        createdAt: c.createdAt,
      }));
      res.json({ chunks: lightChunks, total: lightChunks.length });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // ----------------------------------------------------
  // Protected Semantic Search & Vector Studio Routes
  // ----------------------------------------------------
  app.post('/api/vector/search', authMiddleware, async (req: AuthenticatedRequest, res) => {
    const startTime = Date.now();
    try {
      const userId = req.user!.id;
      const {
        query,
        topK = 4,
        similarityThreshold = 0.3,
        categoryFilter,
        documentIds,
      } = req.body;

      if (!query || typeof query !== 'string') {
        return res.status(400).json({ error: 'Vui lòng nhập từ khóa hoặc câu truy vấn ngữ nghĩa.' });
      }

      const searchRes = await searchVectorStore(
        userId,
        query,
        Number(topK),
        Number(similarityThreshold),
        categoryFilter,
        documentIds
      );

      const executionTimeMs = Date.now() - startTime;

      const formattedResults = searchRes.results.map(item => ({
        chunk: {
          id: item.chunk.id,
          documentId: item.chunk.documentId,
          documentTitle: item.chunk.documentTitle,
          category: item.chunk.category,
          chunkIndex: item.chunk.chunkIndex,
          content: item.chunk.content,
          tokenCount: item.chunk.tokenCount,
          characterCount: item.chunk.characterCount,
          embeddingDim: item.chunk.embedding.length,
          pcaCoords: item.pcaCoords,
          createdAt: item.chunk.createdAt,
        },
        similarity: Math.round(item.similarity * 1000) / 1000,
      }));

      res.json({
        query,
        queryPca: searchRes.queryPca,
        results: formattedResults,
        totalChunksSearched: searchRes.totalChunksSearched,
        executionTimeMs,
        modelUsed: searchRes.modelUsed,
      });
    } catch (err: any) {
      console.error('Vector search error:', err);
      res.status(500).json({ error: err.message || 'Lỗi khi thực hiện tìm kiếm ngữ nghĩa' });
    }
  });

  // ----------------------------------------------------
  // Protected RAG Generation Route
  // ----------------------------------------------------
  app.post('/api/rag/chat', authMiddleware, async (req: AuthenticatedRequest, res) => {
    try {
      const userId = req.user!.id;
      const {
        query,
        topK = 4,
        similarityThreshold = 0.35,
        strictGrounding = true,
        temperature = 0.3,
        categoryFilter,
      } = req.body;

      if (!query || typeof query !== 'string') {
        return res.status(400).json({ error: 'Câu hỏi không được để trống.' });
      }

      const result = await generateRAGAnswer(userId, query, {
        topK: Number(topK),
        similarityThreshold: Number(similarityThreshold),
        strictGrounding: Boolean(strictGrounding),
        temperature: Number(temperature),
        categoryFilter,
      });

      res.json(result);
    } catch (err: any) {
      console.error('RAG chat error:', err);
      res.status(500).json({ error: err.message || 'Lỗi khi xử lý RAG' });
    }
  });

  // ----------------------------------------------------
  // Vite Integration for Dev / Static for Prod
  // ----------------------------------------------------
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Vector RAG Studio server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
