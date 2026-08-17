```dataviewjs
// Укажите путь к вашей папке (оставьте "", если нужна вся база)
const folder = "text"; 

const pages = dv.pages(`"${folder}"`);
const listData = [];

for (let page of pages) {
    // Получаем TFile объект для чтения содержимого
    const file = app.vault.getAbstractFileByPath(page.file.path);
    const content = await app.vault.read(file);
    
    // Ищем первый заголовок # Название через регулярное выражение
    const match = content.match(/^#\s+(.+)\$/m);
    const title = match ? match[1].trim() : page.file.name;
    
    // Добавляем ссылку с красивым отображением заголовка
    listData.push(dv.fileLink(page.file.path, false, title));
}

// Выводим маркированный список, отсортированный по алфавиту
dv.list(listData.sort());
```



