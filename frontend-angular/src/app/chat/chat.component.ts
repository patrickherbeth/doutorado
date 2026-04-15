import { Component } from '@angular/core';
import { KellerService, ConsultaResponse } from '../services/keller.service';

interface HistoricoItem {
 titulo: string;
 pergunta: string;
 data: Date;
}

interface Mensagem {
 tipo: 'user' | 'assistant';
 pergunta?: string;
 resposta?: ConsultaResponse;
 texto?: string;
}

@Component({
 selector: 'app-chat',
 templateUrl: './chat.component.html',
 styleUrls: ['./chat.component.css']
})
export class ChatComponent {
 pergunta = '';
 carregando = false;

 historico: HistoricoItem[] = [];
 mensagens: Mensagem[] = [
  {
   tipo: 'assistant',
   texto: 'Olá. Envie uma consulta jurídica para eu gerar subfatos e ranking dos casos.'
  }
 ];

 constructor(private kellerService: KellerService) {}

 enviarPergunta(): void {
  const texto = this.pergunta.trim();
  if (!texto || this.carregando) {
   return;
  }

  this.mensagens.push({
   tipo: 'user',
   texto
  });

  this.historico.unshift({
   titulo: texto.length > 40 ? texto.slice(0, 40) + '...' : texto,
   pergunta: texto,
   data: new Date()
  });

  this.carregando = true;
  this.pergunta = '';

  this.kellerService.consultar(texto).subscribe({
   next: (res) => {
    this.mensagens.push({
     tipo: 'assistant',
     resposta: res
    });
    this.carregando = false;
   },
   error: (err) => {
    const detalhe =
        err?.error?.detail ||
        err?.error?.mensagem ||
        'Erro ao consultar o endpoint.';
    this.mensagens.push({
     tipo: 'assistant',
     texto: detalhe
    });
    this.carregando = false;
   }
  });
 }

 repetirPesquisa(item: HistoricoItem): void {
  this.pergunta = item.pergunta;
 }

 enviarComEnter(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey) {
   event.preventDefault();
   this.enviarPergunta();
  }
 }
}