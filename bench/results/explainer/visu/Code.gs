/**
 * VISU · Anatomia de um Circuito Visual: Apps Script web app.
 *
 * Ficheiros do projecto:  Code.gs (este)  +  index.html (o conteúdo de visu-anatomia.html, na íntegra)
 * Implementar:  Deploy > New deployment > Web app · Execute as: Me · Who has access: Anyone
 * No Google Sites:  Insert > Apps Script > escolher esta implementação; dar ao bloco ~820 px de altura.
 * Actualizar:  colar o novo index.html, Deploy > Manage deployments > editar a implementação existente >
 *              New version. O URL mantém-se, o bloco no Sites actualiza sozinho.
 */
function doGet() {
  return HtmlService.createHtmlOutputFromFile('index')
    .setTitle('Anatomia de um Circuito Visual · VISU')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL)
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}
