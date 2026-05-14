import { z } from "zod";

// Zod-driven schema inference for strict boundary validation
export const GPTConfigSchema = z.object({
    vocab_size: z.number().int().positive(),
    block_size: z.number().int().positive(),
    n_layer: z.number().int().positive(),
    n_head: z.number().int().positive(),
    n_embd: z.number().int().positive(),
});

export type GPTConfig = z.infer<typeof GPTConfigSchema>;

export class CharTokenizer {
    private stoi: Map<string, number> = new Map();
    private itos: Map<number, string> = new Map();
    public vocab_size: number;

    constructor(source: string | string[]) {
        const chars = typeof source === 'string' 
            ? Array.from(new Set(source.split(''))).sort()
            : source;
            
        this.vocab_size = chars.length;

        chars.forEach((c, i) => {
            this.stoi.set(c, i);
            this.itos.set(i, c);
        });
    }

    public encode(s: string): number[] {
        return s.split('').map((c) => {
            const id = this.stoi.get(c);
            return id ?? 0; // Fallback to 0 if unknown
        });
    }

    public decode(ids: number[]): string {
        return ids.map((i) => {
            const char = this.itos.get(i);
            return char ?? '';
        }).join('');
    }

    public getChar(id: number): string {
        return this.itos.get(id) ?? '';
    }
}
