// src/hooks.server.js

export async function handle({ event, resolve }) {
    const response = await resolve(event);
    
    // enable shared array buffer
    response.headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    response.headers.set('Cross-Origin-Embedder-Policy', 'require-corp');

    // cache control
    // response.headers.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    // response.headers.set('Pragma', 'no-cache');
    // response.headers.set('Expires', '0');
    // content type
    // response.headers.set('Content-Type', 'text/plain');
    
    return response;
}
