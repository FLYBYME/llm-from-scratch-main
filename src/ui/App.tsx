import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { CharTokenizer, type GPTConfig } from '../tokenizer';
import { RangeLayer } from '../model';
import * as tfvis from '@tensorflow/tfjs-vis';
import { Play, Square, Settings, Brain, MessageSquare, Activity, BarChart2 } from 'lucide-react';

// Register custom layer for deserialization in browser
try {
  tf.serialization.registerClass(RangeLayer);
} catch (e) {
  // Already registered
}

interface Prediction {
  char: string;
  prob: number;
}

const App: React.FC = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [tokenizer, setTokenizer] = useState<CharTokenizer | null>(null);
  const [config, setConfig] = useState<GPTConfig | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLoss, setTrainingLoss] = useState<number | null>(null);
  const [trainingStep, setTrainingStep] = useState(0);
  
  const [outputText, setOutputText] = useState('');
  const [prompt, setPrompt] = useState('ROMEO:');
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(100);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [context, setContext] = useState<string[]>([]);
  
  const stopRef = useRef(false);
  const stopTrainRef = useRef(false);

  // Load model and tokenizer on mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('/model/model.json');
        const tokRes = await fetch('/model/tokenizer.json');
        const { chars } = await tokRes.json();
        
        const tok = new CharTokenizer(chars);
        setTokenizer(tok);
        setModel(loadedModel);
        
        // Extract config from model input shape
        const input0 = loadedModel.inputs[0];
        const inputShape = input0 ? input0.shape : null;
        const blockSize = inputShape ? (inputShape[1] as number) : 64;
        
        setConfig({
          vocab_size: chars.length,
          block_size: blockSize,
          n_layer: 4, // Hidden in compiled model usually
          n_head: 4,
          n_embd: 128
        });
        
        console.log('Model loaded successfully');
      } catch (err) {
        console.error('Failed to load model:', err);
      }
    };
    loadModel();
  }, []);

  const toggleVisor = () => {
    tfvis.visor().toggle();
  };

  const trainModel = async () => {
    if (!config || !model || !tokenizer) return;
    setIsTraining(true);
    stopTrainRef.current = false;
    setTrainingStep(0);
    setTrainingLoss(null);

    // Open visor and prepare surface
    const visor = tfvis.visor();
    if (!visor.isOpen()) visor.toggle();
    const history: any[] = [];

    try {
      const res = await fetch('/data/shakespeare.txt');
      const textData = await res.text();
      const tokens = tokenizer.encode(textData);
      
      const optimizer = tf.train.adam(1e-3);
      model.compile({
          optimizer: optimizer,
          loss: tf.losses.softmaxCrossEntropy,
      });

      const batchSize = 32;
      const maxSteps = 200;

      const getBatch = () => {
          return tf.tidy(() => {
              const xBatch: number[][] = [];
              const yBatch: number[][] = [];

              for (let i = 0; i < batchSize; i++) {
                  const ix = Math.floor(Math.random() * (tokens.length - config.block_size - 1));
                  xBatch.push(tokens.slice(ix, ix + config.block_size));
                  yBatch.push(tokens.slice(ix + 1, ix + config.block_size + 1)); 
              }

              const xTensor = tf.tensor2d(xBatch, [batchSize, config.block_size], 'int32');
              const yTensor = tf.oneHot(tf.tensor2d(yBatch, [batchSize, config.block_size], 'int32'), config.vocab_size);
              
              return { x: xTensor, y: yTensor as tf.Tensor3D };
          });
      };

      for (let step = 0; step < maxSteps; step++) {
          if (stopTrainRef.current) break;

          const batch = getBatch();
          
          const fitRes = await model.fit(batch.x, batch.y, {
              batchSize: batchSize,
              epochs: 1,
              verbose: 0
          });

          tf.dispose([batch.x, batch.y]);

          const losses = fitRes.history['loss'];
          const loss = losses?.[0];
          if (typeof loss === 'number') {
              setTrainingLoss(loss);
              history.push({ x: step, y: loss });
              
              // Update live chart
              await tfvis.show.history(
                { name: 'Training Progress', tab: 'Metrics' },
                { history: { loss: history.map(h => h.y) } },
                ['loss']
              );
          }
          setTrainingStep(step + 1);

          // Yield to main thread so UI updates
          await new Promise(r => setTimeout(r, 10));
      }
    } catch (err) {
      console.error("Training error:", err);
    }
    
    setIsTraining(false);
  };

  const generate = async () => {
    if (!model || !tokenizer || !config) return;
    
    setIsGenerating(true);
    stopRef.current = false;
    let currentSeq = tokenizer.encode(prompt);
    setOutputText(prompt);
    
    for (let i = 0; i < maxTokens; i++) {
      if (stopRef.current) break;

      const seqLength = currentSeq.length;
      const truncated = seqLength > config.block_size 
        ? currentSeq.slice(seqLength - config.block_size) 
        : currentSeq;

      const padded = truncated.length < config.block_size
        ? [...new Array(config.block_size - truncated.length).fill(0), ...truncated]
        : truncated;

      setContext(tokenizer.decode(padded).split(''));

      const { nextId, topPredictions } = tf.tidy(() => {
        const input = tf.tensor2d([padded], [1, config.block_size], 'int32');
        const logits = model.predict(input) as tf.Tensor3D;
        
        const finalLogits = logits.slice([0, config.block_size - 1, 0], [1, 1, config.vocab_size])
                                  .squeeze([0, 1]) as tf.Tensor1D;
        
        const scaled = finalLogits.div(tf.scalar(temperature));
        const probs = tf.softmax(scaled);
        
        const pValues = probs.dataSync();
        const topK = Array.from(pValues)
          .map((p, idx) => ({ char: tokenizer.getChar(idx), prob: p }))
          .sort((a, b) => b.prob - a.prob)
          .slice(0, 5);

        const id = tf.multinomial(probs as tf.Tensor1D, 1).dataSync()[0];
        return { nextId: id, topPredictions: topK };
      });

      if (nextId !== undefined) {
        currentSeq.push(nextId);
        setOutputText(prev => prev + tokenizer.getChar(nextId));
      }
      
      setPredictions(topPredictions);
      await new Promise(r => setTimeout(r, 50));
    }
    
    setIsGenerating(false);
  };

  const stopGeneration = () => {
    stopRef.current = true;
    setIsGenerating(false);
  };

  const stopTraining = () => {
    stopTrainRef.current = true;
    setIsTraining(false);
  };

  return (
    <div className="min-h-screen p-8 max-w-5xl mx-auto">
      <header className="mb-12">
        <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
          <Brain className="text-blue-500" /> TypeScript GPT Playground
        </h1>
        <p className="text-slate-400">Visualizing the inner workings of a character-level Transformer model.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="space-y-6">
          {/* Training Controls */}
          <section className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-xl">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-4 flex items-center gap-2">
              <Activity size={16} /> In-Browser Training
            </h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-xs text-slate-400">Step: <span className="text-white font-mono">{trainingStep} / 200</span></div>
                <div className="text-xs text-slate-400">Loss: <span className="text-blue-400 font-mono">{trainingLoss !== null ? trainingLoss.toFixed(4) : '---'}</span></div>
              </div>
              <button
                onClick={isTraining ? stopTraining : trainModel}
                disabled={!model || isGenerating}
                className={`w-full py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all ${
                  isTraining 
                  ? 'bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20' 
                  : 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-lg shadow-emerald-500/20 disabled:opacity-50'
                }`}
              >
                {isTraining ? <><Square size={18} fill="currentColor" /> Stop Training</> : <><Play size={18} fill="currentColor" /> Start Fine-Tuning</>}
              </button>

              <button
                onClick={toggleVisor}
                className="w-full py-2 rounded-lg text-slate-400 text-xs font-medium border border-slate-800 hover:bg-slate-800 flex items-center justify-center gap-2 transition-all"
              >
                <BarChart2 size={14} /> Toggle Live Metrics
              </button>
            </div>
          </section>

          {/* Generation Controls */}
          <section className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-xl">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-4 flex items-center gap-2">
              <Settings size={16} /> Inference Settings
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Temperature ({temperature})</label>
                <input 
                  type="range" min="0.1" max="2.0" step="0.1" 
                  value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-blue-500 bg-slate-800 rounded-lg h-2"
                />
              </div>
              
              <div>
                <label className="block text-xs text-slate-400 mb-1">Max New Tokens ({maxTokens})</label>
                <input 
                  type="range" min="10" max="500" step="10" 
                  value={maxTokens} onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-blue-500 bg-slate-800 rounded-lg h-2"
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-1">Prompt</label>
                <input 
                  type="text" 
                  value={prompt} onChange={(e) => setPrompt(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 p-2 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>

              <button
                onClick={isGenerating ? stopGeneration : generate}
                disabled={!model || isTraining}
                className={`w-full py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all ${
                  isGenerating 
                  ? 'bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20' 
                  : 'bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-500/20 disabled:opacity-50'
                }`}
              >
                {isGenerating ? <><Square size={18} fill="currentColor" /> Stop Generation</> : <><Play size={18} fill="currentColor" /> Generate Text</>}
              </button>
            </div>
          </section>

          <section className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-xl">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-4 flex items-center gap-2">
              <MessageSquare size={16} /> Probabilities
            </h2>
            <div className="space-y-3">
              {predictions.map((p, i) => (
                <div key={i} className="relative h-8 bg-slate-950 rounded border border-slate-800 overflow-hidden flex items-center px-3">
                  <div 
                    className="absolute inset-0 bg-blue-600/20 transition-all duration-300" 
                    style={{ width: `${p.prob * 100}%` }} 
                  />
                  <span className="relative z-10 font-mono text-sm w-8">{p.char === ' ' ? '␣' : p.char === '\n' ? '↵' : p.char}</span>
                  <span className="relative z-10 text-xs text-slate-400 ml-auto">{(p.prob * 100).toFixed(1)}%</span>
                </div>
              ))}
              {predictions.length === 0 && <p className="text-xs text-slate-600 italic">Predictions will appear here...</p>}
            </div>
          </section>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <section className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-xl flex-1 flex flex-col min-h-[400px]">
             <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-4">Generated Text</h2>
             <div className="flex-1 font-mono text-lg leading-relaxed whitespace-pre-wrap text-slate-200">
               {outputText}
               {isGenerating && <span className="inline-block w-2 h-5 bg-blue-500 ml-1 animate-pulse align-middle" />}
             </div>
          </section>

          <section className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-xl">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-4">Context Window ({config?.block_size || 0} tokens)</h2>
            <div className="flex flex-wrap gap-1">
              {context.map((char, i) => (
                <div 
                  key={i} 
                  className={`w-6 h-8 flex items-center justify-center text-xs font-mono rounded border transition-all ${
                    char === ' ' ? 'border-slate-800' : 'border-slate-700 bg-slate-950 text-blue-400'
                  }`}
                  title={`Index: ${i}`}
                >
                  {char === '\n' ? '↵' : char}
                </div>
              ))}
              {context.length === 0 && <p className="text-xs text-slate-600 italic">Context window visualization will appear here...</p>}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default App;
