import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface RankingItem {
    id: number;
    processo: string;
    score: number;
    texto: string;
    subfatos: string[];
}

export interface ConsultaResponse {
    pergunta: string;
    subfatos: string[];
    ranking: RankingItem[];
}

@Injectable({
    providedIn: 'root'
})
export class KellerService {

    private readonly apiUrl = 'http://localhost:8000/api/keller/consulta';

    constructor(private http: HttpClient) {}

    consultar(pergunta: string): Observable<ConsultaResponse> {
        return this.http.post<ConsultaResponse>(this.apiUrl, { pergunta });
    }
}